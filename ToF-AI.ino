#include "head.h"

VL53L5CX_ResultsData measurementData; // Result data class structure, 1356 byes of RAM
int8_t data[8][8][2] = {0};

void setup()
{ 
  bool status = false;
  do
  {
    Serial.begin(115200);

    Serial.println("[*] Start");
    if(tof::setup())
      break;
    Serial.println("[+] SparkFun VL53L5CX set up done");
    if(ml::setup())
      break;
    Serial.println("[+] ML set up done");
    status = true;
  } while (false);
  if (status == false) {
    Serial.println("[-] Boot failed!");
    while (1)
      ;
  }

  return;
}

void loop()
{
  if (tof::myImager.isDataReady() == true)
  {
    tof::parseData(&measurementData, data);
    int predicted_class = ml::runInference(data);
    if (predicted_class >= 0) {
      Serial.print("[+] Predicted: ");
      Serial.println(ml::gesture_labels[predicted_class]);
    }else{
      Serial.println("[-] Predict failed");
    }
    delay(1000);
  }else{
    delay(5);
  }

}
