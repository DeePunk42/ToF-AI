#ifndef TOF_H_
#define TOF_H_

#include "head.h"


namespace tof {
    extern SparkFun_VL53L5CX myImager;
    extern int imageWidth;      // Used to pretty print output

    int setup();
    int parseData(VL53L5CX_ResultsData *measurementData, int8_t data[8][8][2]);
}

#endif