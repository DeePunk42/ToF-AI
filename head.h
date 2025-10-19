#ifndef HEAD_H_
#define HEAD_H_

#include <Wire.h>
#include <SparkFun_VL53L5CX_Library.h> 
#include <stdint.h>
#include <TensorFlowLite.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "ToF.h"
#include "model.h"
#include "head.h"
#include "ml.h"


#endif