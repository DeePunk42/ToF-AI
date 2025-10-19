"""
TFLite Model Conversion Script for Arduino
Converts CNN2D_ST_HandPosture Keras model to quantized TFLite format

Model Information:
- Input: (8, 8, 2) - ToF sensor data (distance, signal per spad)
- Output: 8 classes hand posture recognition
- Classes: None, Like, Dislike, FlatHand, Fist, Love, BreakTime, CrossHands
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path
import glob


class TFLiteConverter:
    def __init__(self, model_path, dataset_path):
        """
        Initialize the TFLite converter

        Args:
            model_path: Path to the Keras .h5 model file
            dataset_path: Path to the dataset directory for representative data
        """
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.model = None

    def load_model(self):
        """Load the Keras model"""
        print(f"Loading model from: {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path)
        print("\nModel Architecture:")
        self.model.summary()
        return self.model

    def load_representative_dataset(self, num_samples=100):
        """
        Load representative dataset from NPZ files for quantization

        Args:
            num_samples: Number of samples to use for calibration

        Returns:
            List of numpy arrays containing representative data
        """
        print(f"\nLoading representative dataset from: {self.dataset_path}")

        # Find all NPZ files in the dataset
        npz_files = []
        for class_dir in Path(self.dataset_path).iterdir():
            if class_dir.is_dir():
                pattern = str(class_dir / "**" / "*.npz")
                npz_files.extend(glob.glob(pattern, recursive=True))

        # Filter out Zone.Identifier files
        npz_files = [f for f in npz_files if not f.endswith('.npz:Zone.Identifier')]

        print(f"Found {len(npz_files)} NPZ files")

        # Load samples
        samples = []
        for npz_file in npz_files[:num_samples]:
            try:
                data = np.load(npz_file)
                # NPZ files contain ToF data - extract distance and signal per spad
                # Assuming the data structure contains these fields
                if 'distance_mm' in data and 'signal_per_spad' in data:
                    distance = data['distance_mm']
                    signal = data['signal_per_spad']

                    # Combine distance and signal into (8, 8, 2) format
                    combined = np.stack([distance, signal], axis=-1)
                    samples.append(combined)
                elif 'arr_0' in data:
                    # Alternative: data might be pre-combined
                    samples.append(data['arr_0'])
            except Exception as e:
                print(f"Warning: Could not load {npz_file}: {e}")
                continue

        if len(samples) == 0:
            print("Warning: No samples loaded. Using synthetic data for calibration.")
            # Generate synthetic data as fallback
            samples = [np.random.randn(8, 8, 2).astype(np.float32) for _ in range(num_samples)]

        print(f"Loaded {len(samples)} samples for quantization calibration")
        return samples

    def representative_data_gen(self, samples):
        """
        Generator function for representative dataset
        Required for full integer quantization
        """
        for sample in samples:
            # Ensure correct shape and type
            sample = np.expand_dims(sample, axis=0).astype(np.float32)
            yield [sample]

    def convert_float32(self, output_path):
        """
        Convert to TFLite with FLOAT32 (no quantization)
        Best accuracy, but larger model size
        """
        print("\n" + "="*60)
        print("Converting to FLOAT32 TFLite model...")
        print("="*60)

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = []

        tflite_model = converter.convert()

        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        file_size = os.path.getsize(output_path) / 1024
        print(f"FLOAT32 model saved to: {output_path}")
        print(f"Model size: {file_size:.2f} KB")

        return tflite_model

    def convert_float16(self, output_path):
        """
        Convert to TFLite with FLOAT16 quantization
        Good balance between size and accuracy
        """
        print("\n" + "="*60)
        print("Converting to FLOAT16 quantized TFLite model...")
        print("="*60)

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

        tflite_model = converter.convert()

        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        file_size = os.path.getsize(output_path) / 1024
        print(f"FLOAT16 model saved to: {output_path}")
        print(f"Model size: {file_size:.2f} KB")

        return tflite_model

    def convert_int8(self, output_path, representative_samples):
        """
        Convert to TFLite with INT8 quantization (dynamic range)
        Smaller size, suitable for microcontrollers
        """
        print("\n" + "="*60)
        print("Converting to INT8 quantized TFLite model (dynamic range)...")
        print("="*60)

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: self.representative_data_gen(representative_samples)

        tflite_model = converter.convert()

        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        file_size = os.path.getsize(output_path) / 1024
        print(f"INT8 model saved to: {output_path}")
        print(f"Model size: {file_size:.2f} KB")

        return tflite_model

    def convert_int8_full(self, output_path, representative_samples):
        """
        Convert to TFLite with full INT8 quantization
        Smallest size, best for microcontrollers with limited resources
        Input and output are also INT8
        """
        print("\n" + "="*60)
        print("Converting to FULL INT8 quantized TFLite model...")
        print("="*60)

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: self.representative_data_gen(representative_samples)

        # Enforce full integer quantization
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        tflite_model = converter.convert()

        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        file_size = os.path.getsize(output_path) / 1024
        print(f"FULL INT8 model saved to: {output_path}")
        print(f"Model size: {file_size:.2f} KB")

        return tflite_model

    def generate_c_header(self, tflite_model, output_path, model_name="hand_posture_model"):
        """
        Generate C header file for Arduino
        Converts TFLite model to a C byte array
        """
        print(f"\nGenerating C header file: {output_path}")

        with open(output_path, 'w') as f:
            f.write(f"// Auto-generated TFLite model for Arduino\n")
            f.write(f"// Model: CNN2D_ST_HandPosture_8classes\n")
            f.write(f"// Input: (8, 8, 2) - ToF sensor data\n")
            f.write(f"// Output: 8 classes\n\n")
            f.write(f"#ifndef {model_name.upper()}_H\n")
            f.write(f"#define {model_name.upper()}_H\n\n")
            f.write(f"const unsigned int {model_name}_len = {len(tflite_model)};\n")
            f.write(f"const unsigned char {model_name}[] = {{\n")

            # Write bytes in rows of 12
            for i in range(0, len(tflite_model), 12):
                chunk = tflite_model[i:i+12]
                hex_values = ', '.join([f'0x{b:02x}' for b in chunk])
                f.write(f"  {hex_values},\n")

            f.write(f"}};\n\n")
            f.write(f"#endif // {model_name.upper()}_H\n")

        print(f"C header file saved to: {output_path}")

    def analyze_model(self, tflite_path):
        """Analyze the converted TFLite model"""
        print(f"\n" + "="*60)
        print(f"Analyzing TFLite model: {tflite_path}")
        print("="*60)

        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print("\nInput Details:")
        for detail in input_details:
            print(f"  Name: {detail['name']}")
            print(f"  Shape: {detail['shape']}")
            print(f"  Type: {detail['dtype']}")
            if 'quantization_parameters' in detail:
                quant = detail['quantization_parameters']
                if quant['scales'].size > 0:
                    print(f"  Quantization - Scale: {quant['scales']}, Zero point: {quant['zero_points']}")

        print("\nOutput Details:")
        for detail in output_details:
            print(f"  Name: {detail['name']}")
            print(f"  Shape: {detail['shape']}")
            print(f"  Type: {detail['dtype']}")
            if 'quantization_parameters' in detail:
                quant = detail['quantization_parameters']
                if quant['scales'].size > 0:
                    print(f"  Quantization - Scale: {quant['scales']}, Zero point: {quant['zero_points']}")

        # Get tensor details
        tensor_details = interpreter.get_tensor_details()
        print(f"\nTotal tensors: {len(tensor_details)}")

        return interpreter


def main():
    """Main conversion workflow"""

    # Configuration
    MODEL_PATH = "model/CNN2D_ST_HandPosture_8classes.h5"
    DATASET_PATH = "model/datasets/ST_VL53L8CX_handposture_dataset"
    OUTPUT_DIR = "model/tflite"

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*60)
    print("TFLite Model Converter for Arduino")
    print("CNN2D_ST_HandPosture Hand Posture Recognition")
    print("="*60)

    # Initialize converter
    converter = TFLiteConverter(MODEL_PATH, DATASET_PATH)

    # Load Keras model
    converter.load_model()

    # Load representative dataset for quantization
    representative_samples = converter.load_representative_dataset(num_samples=100)

    # Convert to different formats
    models = {}

    # 1. FLOAT32 (baseline, no quantization)
    models['float32'] = converter.convert_float32(
        os.path.join(OUTPUT_DIR, "hand_posture_float32.tflite")
    )

    # 2. FLOAT16 (recommended for most Arduino boards)
    models['float16'] = converter.convert_float16(
        os.path.join(OUTPUT_DIR, "hand_posture_float16.tflite")
    )

    # 3. INT8 dynamic range (good compromise)
    models['int8'] = converter.convert_int8(
        os.path.join(OUTPUT_DIR, "hand_posture_int8.tflite"),
        representative_samples
    )

    # 4. FULL INT8 (smallest, best for resource-constrained devices)
    models['int8_full'] = converter.convert_int8_full(
        os.path.join(OUTPUT_DIR, "hand_posture_int8_full.tflite"),
        representative_samples
    )

    # Generate C header files for Arduino
    print("\n" + "="*60)
    print("Generating C header files for Arduino...")
    print("="*60)

    converter.generate_c_header(
        models['float16'],
        os.path.join(OUTPUT_DIR, "hand_posture_float16.h"),
        "hand_posture_model_float16"
    )

    converter.generate_c_header(
        models['int8_full'],
        os.path.join(OUTPUT_DIR, "hand_posture_int8_full.h"),
        "hand_posture_model_int8"
    )

    # Analyze models
    print("\n" + "="*60)
    print("Model Analysis")
    print("="*60)

    for name, path in [
        ("FLOAT16", os.path.join(OUTPUT_DIR, "hand_posture_float16.tflite")),
        ("INT8 FULL", os.path.join(OUTPUT_DIR, "hand_posture_int8_full.tflite"))
    ]:
        converter.analyze_model(path)

    # Summary
    print("\n" + "="*60)
    print("Conversion Summary")
    print("="*60)
    print("\nGenerated Files:")
    print(f"  Output Directory: {OUTPUT_DIR}/")
    print("\nTFLite Models:")
    print(f"  1. hand_posture_float32.tflite    - FLOAT32 (baseline)")
    print(f"  2. hand_posture_float16.tflite    - FLOAT16 (recommended)")
    print(f"  3. hand_posture_int8.tflite       - INT8 dynamic range")
    print(f"  4. hand_posture_int8_full.tflite  - INT8 full (smallest)")
    print("\nArduino Header Files:")
    print(f"  1. hand_posture_float16.h         - FLOAT16 for Arduino")
    print(f"  2. hand_posture_int8_full.h       - INT8 FULL for Arduino")

    print("\n" + "="*60)
    print("Recommendation for Arduino:")
    print("="*60)
    print("For most Arduino boards (e.g., Arduino Nano 33 BLE Sense):")
    print("  - Use: hand_posture_int8_full.tflite or hand_posture_int8_full.h")
    print("  - This provides the smallest model size with full INT8 quantization")
    print("\nFor Arduino with more resources:")
    print("  - Use: hand_posture_float16.tflite or hand_posture_float16.h")
    print("  - Better accuracy with moderate size increase")
    print("="*60)

    print("\nConversion completed successfully!")


if __name__ == "__main__":
    main()
