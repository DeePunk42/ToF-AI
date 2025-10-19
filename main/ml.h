#ifndef ML_H_
#define ML_H_

#include "head.h"

namespace ml {
    extern const tflite::Model* model;
    extern const char* gesture_labels[8];

    int setup();
    int runInference(int8_t data[8][8][2]);
    int8_t quantize_float_to_int8(float value);
}

#endif