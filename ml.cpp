#include "ml.h"

namespace ml {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  constexpr int kTensorArenaSize = 20 * 1024;
  alignas(16) uint8_t tensor_arena[kTensorArenaSize];

  const char* gesture_labels[8] = {
    "BreakTime",
    "CrossHands", 
    "Dislike",
    "Fist",
    "FlatHand",
    "Like",
    "Love",
    "None"
  };

  int setup() {
      tflite::InitializeTarget();

      model = tflite::GetModel(g_model_data);
      if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("[-] Model provided is schema version not equal");
        return -1;
      }
      Serial.println("[+] Model loaded");

      static tflite::AllOpsResolver resolver;

      static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
      interpreter = &static_interpreter;

      TfLiteStatus allocate_status = interpreter->AllocateTensors();
      if (allocate_status != kTfLiteOk) {
        Serial.println("[-] tensor allocated failed!");
        return -1;
      }
      Serial.println("[+] Tensors allocated");

      input = interpreter->input(0);
      output = interpreter->output(0);
      
      Serial.print("[*] Input shape: ");
      Serial.print(input->dims->data[1]); Serial.print("x");
      Serial.print(input->dims->data[2]); Serial.print("x");
      Serial.println(input->dims->data[3]);
      
      Serial.print("[*] Input type: ");
      Serial.println(input->type == kTfLiteInt8 ? "int8" : "float32");

      Serial.print("[*] Input scale: ");
      Serial.println(input->params.scale);
      
      Serial.print("[*] Input zero point: ");
      Serial.print(input->params.zero_point, 6);
      Serial.println("");
      
      Serial.print("[*] Output shape: ");
      Serial.println(output->dims->data[1]);
      
      Serial.print("[*] Output type: ");
      Serial.println(output->type == kTfLiteInt8 ? "int8" : "float32");
      
      Serial.print("[*] Arena used bytes: ");
      Serial.println(interpreter->arena_used_bytes());
      
      Serial.println("\n[*] Ready for inference!");

      return 0;
  }

  int runInference(int8_t data[8][8][2]) {
    int8_t* input_data = input->data.int8;

    unsigned long start_time = micros();
    TfLiteStatus invoke_status = interpreter->Invoke();
    unsigned long end_time = micros();
    
    if (invoke_status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
      return -1;
    }
    
    Serial.print("Inference time: ");
    Serial.print(end_time - start_time);
    Serial.println(" us");

    int predicted_class = -1;
    float max_score = -1000.0f;
  
    // Int8 量化输出
    float output_scale = output->params.scale;
    int output_zero_point = output->params.zero_point;
    int8_t* output_data = output->data.int8;
    
    Serial.print("Class scores: ");
    for (int i = 0; i < 8; i++) {
      // 反量化公式: x = (q - zero_point) * scale
      float score = (output_data[i] - output_zero_point) * output_scale;
      Serial.print(gesture_labels[i]);
      Serial.print(": ");
      Serial.print(score, 4);
      Serial.print(" ");
      
      if (score > max_score) {
        max_score = score;
        predicted_class = i;
      }
    }
    Serial.println();
    Serial.print("Confidence: ");
    Serial.println(max_score, 4);
    return predicted_class;
  }

  int8_t quantize_float_to_int8(float value) {
    float input_scale = input->params.scale;
    int input_zero_point = input->params.zero_point;

    int32_t q = (int32_t)roundf(value / input_scale) + input_zero_point;
    return (int8_t)constrain(q, -128, 127);
  }
}