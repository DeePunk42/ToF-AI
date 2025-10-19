"""
简化版TFLite模型验证脚本
快速验证量化模型在各个类别上的准确性
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path
import glob


def load_samples_by_class(dataset_path, class_name, max_samples=10):
    """加载指定类别的样本"""
    class_dir = Path(dataset_path) / class_name
    samples = []

    if not class_dir.exists():
        print(f"警告: 类别目录不存在: {class_dir}")
        return samples

    # 查找NPZ文件
    npz_files = glob.glob(str(class_dir / "**" / "*.npz"), recursive=True)
    npz_files = [f for f in npz_files if not f.endswith(':Zone.Identifier')]

    for npz_file in npz_files[:max_samples]:
        try:
            data = np.load(npz_file)

            # 方法1: ST VL53L8CX格式 - zone_data字段
            if 'zone_data' in data and 'zone_head' in data:
                zone_data = data['zone_data']
                zone_head = data['zone_head']

                # 找到distance_mm和signal_per_spad的索引
                distance_idx = np.where(zone_head == 'distance_mm')[0][0]
                signal_idx = np.where(zone_head == 'signal_per_spad')[0][0]

                # 取第一帧数据
                frame = zone_data[0]  # (64, features)

                # 每个字段占用的列数
                cols_per_field = frame.shape[1] // len(zone_head)

                # 提取distance和signal列
                distance_cols = frame[:, distance_idx * cols_per_field : (distance_idx + 1) * cols_per_field]
                signal_cols = frame[:, signal_idx * cols_per_field : (signal_idx + 1) * cols_per_field]

                # 取每行的第一个值
                distance = distance_cols[:, 0] if distance_cols.shape[1] > 0 else distance_cols.flatten()
                signal = signal_cols[:, 0] if signal_cols.shape[1] > 0 else signal_cols.flatten()

                # Reshape到8x8
                distance_8x8 = distance.reshape(8, 8)
                signal_8x8 = signal.reshape(8, 8)

                # 合并为(8, 8, 2)
                combined = np.stack([distance_8x8, signal_8x8], axis=-1)
                samples.append(combined.astype(np.float32))

            # 方法2: 分离的distance和signal字段
            elif 'distance_mm' in data and 'signal_per_spad' in data:
                distance = np.array(data['distance_mm']).reshape(8, 8)
                signal = np.array(data['signal_per_spad']).reshape(8, 8)
                combined = np.stack([distance, signal], axis=-1)
                samples.append(combined.astype(np.float32))

            # 方法3: 单一数组
            elif 'arr_0' in data:
                arr = data['arr_0']
                if arr.shape == (8, 8, 2):
                    samples.append(arr.astype(np.float32))

        except Exception as e:
            print(f"警告: 加载失败 {npz_file}: {e}")

    return samples


def predict_tflite(interpreter, input_details, output_details, data):
    """使用TFLite模型进行预测"""

    # 预处理
    processed = np.expand_dims(data, axis=0)

    # 量化（如果需要）
    if input_details['dtype'] == np.int8:
        quant_params = input_details['quantization_parameters']
        if quant_params['scales'].size > 0:
            scale = quant_params['scales'][0]
            zero_point = quant_params['zero_points'][0]
            processed = processed / scale + zero_point
            processed = np.clip(processed, -128, 127).astype(np.int8)
    else:
        processed = processed.astype(np.float32)

    # 推理
    interpreter.set_tensor(input_details['index'], processed)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details['index'])[0]

    # 反量化
    if output_details['dtype'] == np.int8:
        quant_params = output_details['quantization_parameters']
        if quant_params['scales'].size > 0:
            scale = quant_params['scales'][0]
            zero_point = quant_params['zero_points'][0]
            output = (output.astype(np.float32) - zero_point) * scale

    # Softmax
    exp_output = np.exp(output - np.max(output))
    probabilities = exp_output / exp_output.sum()

    return np.argmax(probabilities), probabilities


def validate_model():
    """验证模型"""

    # 配置
    MODEL_PATH = "model/tflite/hand_posture_int8_full.tflite"
    DATASET_PATH = "model/datasets/ST_VL53L8CX_handposture_dataset"
    CLASS_NAMES = ["None", "Like", "Dislike", "FlatHand", "Fist", "Love", "BreakTime", "CrossHands"]
    SAMPLES_PER_CLASS = 20  # 每个类别测试的样本数

    # 检查文件
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在: {MODEL_PATH}")
        print("请先运行 convert_simple.py 或 convert_to_tflite.py 生成模型")
        return

    if not os.path.exists(DATASET_PATH):
        print(f"错误: 数据集不存在: {DATASET_PATH}")
        return

    # 加载模型
    print("="*60)
    print("TFLite模型验证")
    print("="*60)
    print(f"\n加载模型: {MODEL_PATH}")

    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    print(f"输入: {input_details['shape']} - {input_details['dtype']}")
    print(f"输出: {output_details['shape']} - {output_details['dtype']}")

    # 测试每个类别
    print("\n" + "="*60)
    print("测试各个类别")
    print("="*60)

    total_correct = 0
    total_samples = 0
    class_results = {}

    for class_idx, class_name in enumerate(CLASS_NAMES):
        print(f"\n测试类别: {class_name} (索引 {class_idx})")

        # 加载样本
        samples = load_samples_by_class(DATASET_PATH, class_name, SAMPLES_PER_CLASS)

        if len(samples) == 0:
            print(f"  跳过 (无可用样本)")
            continue

        print(f"  加载了 {len(samples)} 个样本")

        # 测试每个样本
        correct = 0
        predictions_count = {i: 0 for i in range(len(CLASS_NAMES))}

        for sample in samples:
            pred_class, probs = predict_tflite(interpreter, input_details, output_details, sample)
            predictions_count[pred_class] += 1

            if pred_class == class_idx:
                correct += 1

        accuracy = correct / len(samples) if len(samples) > 0 else 0
        total_correct += correct
        total_samples += len(samples)

        # 保存结果
        class_results[class_name] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(samples),
            'predictions': predictions_count
        }

        # 打印结果
        print(f"  准确率: {correct}/{len(samples)} = {accuracy:.2%}")
        print(f"  预测分布:")
        for pred_idx, count in predictions_count.items():
            if count > 0:
                print(f"    {CLASS_NAMES[pred_idx]:12s}: {count:3d} ({count/len(samples):.1%})")

    # 打印总结
    print("\n" + "="*60)
    print("验证总结")
    print("="*60)

    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    print(f"\n总体准确率: {total_correct}/{total_samples} = {overall_accuracy:.2%}")

    print("\n各类别准确率:")
    print(f"{'类别':<12s} {'准确率':>10s} {'正确数':>8s} {'总数':>6s}")
    print("-"*60)

    for class_name in CLASS_NAMES:
        if class_name in class_results:
            result = class_results[class_name]
            print(f"{class_name:<12s} {result['accuracy']:>9.1%} "
                  f"{result['correct']:>8d} {result['total']:>6d}")
        else:
            print(f"{class_name:<12s} {'N/A':>10s} {'N/A':>8s} {'N/A':>6s}")

    print("="*60)

    # 保存结果
    print(f"\n验证完成!")

    if overall_accuracy < 0.8:
        print("\n警告: 准确率较低，可能原因:")
        print("  1. 量化导致精度损失")
        print("  2. 预处理参数不匹配")
        print("  3. 数据集质量问题")
        print("\n建议:")
        print("  - 使用 convert_to_tflite.py 中的真实数据集进行量化校准")
        print("  - 检查预处理步骤是否与训练时一致")
        print("  - 尝试使用 FLOAT16 量化而非 INT8")
    elif overall_accuracy > 0.95:
        print(f"\n优秀! 模型性能良好，可以部署到Arduino")
    else:
        print(f"\n模型性能可接受，可以部署")


if __name__ == "__main__":
    validate_model()
