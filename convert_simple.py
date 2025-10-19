"""
Simplified TFLite Conversion Script
Quick conversion without representative dataset (uses synthetic data)
"""

import os
import numpy as np
import tensorflow as tf


def convert_model_simple():
    """Simple conversion with all quantization options"""

    MODEL_PATH = "model/CNN2D_ST_HandPosture_8classes.h5"
    OUTPUT_DIR = "model/tflite"

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading Keras model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    model.summary()

    # Representative dataset generator (synthetic data)
    def representative_data_gen():
        """Generate synthetic ToF data for quantization calibration"""
        for _ in range(100):
            # Generate random data matching input shape (1, 8, 8, 2)
            # In production, use real ToF sensor data for better quantization
            data = np.random.randn(1, 8, 8, 2).astype(np.float32)
            yield [data]

    # 1. INT8 Full Quantization (Recommended for Arduino)
    print("\n" + "="*50)
    print("Converting to INT8 Full Quantization...")
    print("="*50)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    output_path = os.path.join(OUTPUT_DIR, "hand_posture_int8_full.tflite")
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Model saved: {output_path}")
    print(f"Size: {len(tflite_model) / 1024:.2f} KB")

    # 2. Generate C header for Arduino
    print("\nGenerating C header file...")
    header_path = os.path.join(OUTPUT_DIR, "hand_posture_model.h")

    with open(header_path, 'w') as f:
        f.write("// TFLite model for Arduino - Hand Posture Recognition\n")
        f.write("// Input: (8, 8, 2) INT8 - ToF sensor data\n")
        f.write("// Output: 8 classes INT8\n\n")
        f.write("#ifndef HAND_POSTURE_MODEL_H\n")
        f.write("#define HAND_POSTURE_MODEL_H\n\n")
        f.write(f"const unsigned int hand_posture_model_len = {len(tflite_model)};\n")
        f.write("const unsigned char hand_posture_model[] = {\n")

        for i in range(0, len(tflite_model), 12):
            chunk = tflite_model[i:i+12]
            hex_values = ', '.join([f'0x{b:02x}' for b in chunk])
            f.write(f"  {hex_values},\n")

        f.write("};\n\n")
        f.write("#endif // HAND_POSTURE_MODEL_H\n")

    print(f"Header saved: {header_path}")

    # 3. Model info
    print("\n" + "="*50)
    print("Model Information")
    print("="*50)

    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    print(f"\nInput:")
    print(f"  Shape: {input_details['shape']}")
    print(f"  Type: {input_details['dtype']}")

    print(f"\nOutput:")
    print(f"  Shape: {output_details['shape']}")
    print(f"  Type: {output_details['dtype']}")

    print("\n" + "="*50)
    print("Conversion Complete!")
    print("="*50)
    print(f"\nUse '{header_path}' in your Arduino project")
    print("\nClasses:")
    classes = ["None", "Like", "Dislike", "FlatHand", "Fist", "Love", "BreakTime", "CrossHands"]
    for i, cls in enumerate(classes):
        print(f"  {i}: {cls}")


if __name__ == "__main__":
    convert_model_simple()
