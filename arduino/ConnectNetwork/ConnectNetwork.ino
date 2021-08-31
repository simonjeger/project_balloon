/*
  WiFi Test for Nano 33 IoT

  Connects to Wifi network and prints IP address on serial monitor

*/

#include <SPI.h>
#include <WiFiNINA.h>

char ssid[] = "Feng_2.4G_network";      // Wifi SSID
char pass[] = "auhauhihc";       // Wifi password

int status = WL_IDLE_STATUS;

// Initialize the Wifi client
WiFiSSLClient client;

void setup() {
  //Initialize serial and wait for port to open:
  Serial.begin(9600);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB port only
  }
  // check for the WiFi module:
  if (WiFi.status() == WL_NO_MODULE) {
    Serial.println("Communication with WiFi module failed!");
    // don't continue
    while (true);
  }

  String fv = WiFi.firmwareVersion();
  if (fv < WIFI_FIRMWARE_LATEST_VERSION) {
    Serial.println("Please upgrade the firmware");
  }
  connectToAP();    // Connect to Wifi Access Point
  printWifiStatus();
}

void loop() {}

void printWifiStatus() {
  Serial.print("SSID: ");
  Serial.println(WiFi.SSID());

  IPAddress ip = WiFi.localIP(); // Device IP address
  Serial.print("IP Address: ");
  Serial.println(ip);
}

void connectToAP() {
  // Try to connect to Wifi network
  while ( status != WL_CONNECTED) {
    Serial.print("Attempting to connect to SSID: ");
    Serial.println(ssid);
    // Connect to WPA/WPA2 network
    status = WiFi.begin(ssid, pass);

    // wait 1 second for connection:
    delay(1000);
    Serial.println("Connected...");
  }
}
