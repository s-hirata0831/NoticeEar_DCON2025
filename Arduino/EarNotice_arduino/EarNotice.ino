#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <FastLED.h>

#define SERVICE_UUID        "55725ac1-066c-48b5-8700-2d9fb3603c5e"
#define CHARACTERISTIC_UUID "69ddb59c-d601-4ea4-ba83-44f679a670ba"
#define BLE_DEVICE_NAME     "EarNotice"
#define PIN_LED             21
#define NUM_LEDS            1

enum State{
  common = 1,
  animal = 2,
  emerge = 3
};

CRGB leds[NUM_LEDS];
BLEServer *pServer = NULL;
BLECharacteristic *pCharacteristic = NULL;
bool deviceConnected = false;
bool oldDeviceConnected = false;
bool newDataReceived = false;
String received;
State rxValue;
unsigned long lastSendTime = 0;
const unsigned long sendInterval = 1000;

class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer *pServer) {
        deviceConnected = true;
        Serial.println("Device connected!");
    }

    void onDisconnect(BLEServer *pServer) {
        deviceConnected = false;
        Serial.println("Device disconnected!");
    }
};

class MyCharacteristicCallbacks: public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic *pCharacteristic) {
        String value = pCharacteristic->getValue();
        if (!value.isEmpty()) {
          received = value.c_str();
          newDataReceived = true; // 新しいデータを受信したフラグを立てる
          Serial.print("Received Value: ");
          Serial.println(received);
          // 文字列をState型に変換
        if (received == "common") {
          rxValue = common;
        } else if (received == "animal") {
          rxValue = animal;
        } else if (received == "emerge") {
          rxValue = emerge;
        } else {
          // 予期しない値が来た場合の処理（デフォルト設定）
          rxValue = common;  // またはエラー処理を行う
          Serial.println("Unknown value received");
        }
      }
    }
};

void setup() {
    Serial.begin(9600); // USBシリアルの初期化
    FastLED.addLeds<WS2812B, PIN_LED, GRB>(leds, NUM_LEDS);
    leds[0] = CRGB(40, 40, 40); // 初期LED設定
    FastLED.show();
    pinMode(1,OUTPUT);

    BLEDevice::init(BLE_DEVICE_NAME);
    pServer = BLEDevice::createServer();
    pServer->setCallbacks(new MyServerCallbacks());

    BLEService *pService = pServer->createService(SERVICE_UUID);
    pCharacteristic = pService->createCharacteristic(
        CHARACTERISTIC_UUID,
        BLECharacteristic::PROPERTY_WRITE |
        BLECharacteristic::PROPERTY_NOTIFY |
        BLECharacteristic::PROPERTY_READ
    );
    pCharacteristic->setCallbacks(new MyCharacteristicCallbacks());
    pCharacteristic->addDescriptor(new BLE2902());
    pCharacteristic->setValue("Initial Data");
    pService->start();

    BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(SERVICE_UUID);
    BLEDevice::startAdvertising();
    Serial.println("startAdvertising");
}

void loop() {
    // デバイスの接続状態を確認
    if (!deviceConnected && oldDeviceConnected) {
        delay(500); 
        pServer->startAdvertising();
        Serial.println("restartAdvertising");
        oldDeviceConnected = deviceConnected;
    }

    if (deviceConnected && !oldDeviceConnected) {
        oldDeviceConnected = deviceConnected;
    }

    // 新しいデータを受信した場合の処理
    if (newDataReceived) {
        Serial.print("Processing received data: ");
        Serial.println(rxValue);

        switch(rxValue){
          case common:
            leds[0] = CRGB::Gold;//黄色
            FastLED.show();
            //ブーブー
            digitalWrite(1, HIGH);
            delay(500);
            digitalWrite(1, LOW); 
            delay(100);
            digitalWrite(1, HIGH);
            delay(1000);
            digitalWrite(1, LOW); 
            break;
          case animal:
            leds[0] = CRGB::Green;//緑
            FastLED.show();
            //ブッブッ
            digitalWrite(1, HIGH);
            delay(200);
            digitalWrite(1, LOW); 
            delay(200);   
            digitalWrite(1, HIGH);
            delay(200);
            digitalWrite(1, LOW); 
            delay(200);
            break;
          case emerge:
            leds[0] = CRGB::Red;//赤
            FastLED.show();
            //ブッブーブッブーブッブー
            digitalWrite(1, HIGH);
            delay(100);
            digitalWrite(1, LOW); 
            delay(100);
            digitalWrite(1, HIGH);
            delay(400);
            digitalWrite(1, LOW); 
            delay(200);
            
            digitalWrite(1, HIGH);
            delay(100);
            digitalWrite(1, LOW); 
            delay(100);   
            digitalWrite(1, HIGH);
            delay(400);
            digitalWrite(1, LOW); 
            delay(200);
            
            digitalWrite(1, HIGH);
            delay(100);
            digitalWrite(1, LOW); 
            delay(100);  
            digitalWrite(1, HIGH);
            delay(400);
            digitalWrite(1, LOW);
            break; 
        }
        delay(3000);
        newDataReceived = false; // フラグをリセット
    }

    delay(100); // 必要に応じて適宜調整
}
