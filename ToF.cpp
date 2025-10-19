#include "ToF.h"

namespace tof {
  SparkFun_VL53L5CX myImager;

  int imageResolution = 0; // Used to pretty print output
  int imageWidth = 0;      // Used to pretty print output

  long measurements = 0;         // Used to calculate actual output rate
  long measurementStartTime = 0; // Used to calculate actual output rate


  int setup() {
    Wire.begin(); // This resets I2C bus to 100kHz
    Wire.setClock(1000000); //Sensor has max I2C freq of 1MHz

    myImager.setWireMaxPacketSize(128); // Increase default from 32 bytes to 128 - not supported on all platforms

    Serial.println("[*] Initializing sensor board. This can take up to 10s. Please wait.");
    if (myImager.begin() == false)
    {
      Serial.println("[-] Sensor not found - check your wiring. Freezing");
      return -1;
    }

    myImager.setResolution(8 * 8); // Enable all 64 pads

    imageResolution = myImager.getResolution(); // Query sensor for current resolution - either 4x4 or 8x8
    imageWidth = sqrt(imageResolution);         // Calculate printing width

    // Using 4x4, min frequency is 1Hz and max is 60Hz
    // Using 8x8, min frequency is 1Hz and max is 15Hz
    myImager.setRangingFrequency(15);

    myImager.startRanging();

    measurementStartTime = millis();

    return 0;
  }

  int parseData(VL53L5CX_ResultsData *measurementData, int8_t data[8][8][2]) {

    if (tof::myImager.getRangingData(measurementData)) // Read distance data into array
    {
      for (int y = 0; y <= tof::imageWidth * (tof::imageWidth - 1); y += tof::imageWidth)
      {
        for (int x = tof::imageWidth - 1; x >= 0; x--)
        {
          data[x][y][0] = ml::quantize_float_to_int8(measurementData->distance_mm[x + y] / 4096.0f);
          data[x][y][1] = ml::quantize_float_to_int8(measurementData->signal_per_spad[x + y] / 65535.0f);
        }
      }
      return 0;
    }
    return -1;
  }


}