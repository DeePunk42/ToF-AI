#!/usr/bin/env python3
"""
将Keras模型转换为TensorFlow Lite格式并验证准确性
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import glob
from datetime import datetime


def load_and_preprocess_sample(npz_file):
    """加载并预处理单个样本 - 完全按照STM32代码的处理流程"""
    try:
        data = np.load(npz_file, allow_pickle=True)
        zone_data = data['zone_data']  # (4, 64, N)
        zone_head = data['zone_head']
        
        # 找到distance_mm, signal_per_spad, target_status, valid的索引
        zone_head_list = [str(h) for h in zone_head]
        signal_idx = zone_head_list.index('signal_per_spad')
        distance_idx = zone_head_list.index('distance_mm')
        status_idx = zone_head_list.index('target_status')
        valid_idx = zone_head_list.index('valid')
        
        # 提取数据 (64, N)
        signal = zone_data[signal_idx, :, :]
        distance = zone_data[distance_idx, :, :]
        target_status = zone_data[status_idx, :, :]
        valid = zone_data[valid_idx, :, :]
        
        # 选择中间帧
        frame_idx = signal.shape[1] // 2
        signal_frame = signal[:, frame_idx].copy()  # (64,)
        distance_frame = distance[:, frame_idx].copy()  # (64,)
        status_frame = target_status[:, frame_idx]  # (64,)
        valid_frame = valid[:, frame_idx]  # (64,)
        
        # ===== 预处理步骤1: 帧验证 (ValidateFrame) =====
        # 配置参数 (来自config.yaml)
        MIN_DISTANCE = 100.0
        MAX_DISTANCE = 400.0
        BACKGROUND_REMOVAL = 120.0
        DEFAULT_RANGING_VALUE = 4000.0
        DEFAULT_SIGNAL_VALUE = 0.0
        RANGING_OK_5 = 5
        RANGING_OK_9 = 9
        
        # 找到所有有效区域中的最小距离
        min_distance = 4000.0
        for idx in range(64):
            is_valid_zone = (valid_frame[idx] > 0 and 
                            (status_frame[idx] == RANGING_OK_5 or status_frame[idx] == RANGING_OK_9) and
                            distance_frame[idx] < min_distance)
            if is_valid_zone:
                min_distance = distance_frame[idx]
        
        # 检查帧是否有效 (虽然我们不需要这个标志,但保持一致性)
        is_valid_frame = (min_distance < MAX_DISTANCE and min_distance > MIN_DISTANCE)
        
        # 应用背景移除和默认值填充
        for idx in range(64):
            is_valid_zone = (valid_frame[idx] > 0 and
                            (status_frame[idx] == RANGING_OK_5 or status_frame[idx] == RANGING_OK_9) and
                            distance_frame[idx] < min_distance + BACKGROUND_REMOVAL)
            
            if not is_valid_zone:
                distance_frame[idx] = DEFAULT_RANGING_VALUE
                signal_frame[idx] = DEFAULT_SIGNAL_VALUE
        
        # ===== 预处理步骤2: 归一化 (NormalizeData) =====
        # 归一化参数 (来自app_utils.h)
        NORMALIZATION_RANGING_CENTER = 295.0
        NORMALIZATION_RANGING_IQR = 196.0
        NORMALIZATION_SIGNAL_CENTER = 281.0
        NORMALIZATION_SIGNAL_IQR = 452.0
        
        # 归一化: (value - median) / IQR
        distance_norm = (distance_frame - NORMALIZATION_RANGING_CENTER) / NORMALIZATION_RANGING_IQR
        signal_norm = (signal_frame - NORMALIZATION_SIGNAL_CENTER) / NORMALIZATION_SIGNAL_IQR
        
        # 重塑为(8, 8, 2)
        input_tensor = np.zeros((8, 8, 2), dtype=np.float32)
        input_tensor[:, :, 0] = distance_norm.reshape(8, 8)
        input_tensor[:, :, 1] = signal_norm.reshape(8, 8)
        
        return input_tensor
    except Exception as e:
        print(f"Error loading {npz_file}: {e}")
        return None


def prepare_dataset(dataset_path, class_names, max_samples_per_class=None):
    """准备测试数据集"""
    X = []
    y = []
    
    print(f"\n加载数据集: {dataset_path}")
    print("-" * 80)
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_dir):
            print(f"警告: 类别 '{class_name}' 不存在")
            continue
        
        # 查找所有NPZ文件
        npz_files = glob.glob(os.path.join(class_dir, '**', 'npz', '*.npz'), recursive=True)
        npz_files = [f for f in npz_files if not f.endswith(':Zone.Identifier')]
        
        if max_samples_per_class:
            npz_files = npz_files[:max_samples_per_class]
        
        print(f"  {class_name:15s}: 加载 {len(npz_files)} 个样本...", end=' ')
        
        loaded = 0
        for npz_file in npz_files:
            sample = load_and_preprocess_sample(npz_file)
            if sample is not None:
                X.append(sample)
                y.append(class_idx)
                loaded += 1
        
        print(f"✓ 成功加载 {loaded} 个")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n总样本数: {len(X)}")
    print(f"数据形状: {X.shape}")
    print(f"标签形状: {y.shape}")
    
    return X, y


def convert_to_tflite(model, output_path, quantize=False, representative_dataset=None):
    """转换Keras模型为TFLite"""
    print(f"\n转换模型为TFLite...")
    print(f"量化: {'是' if quantize else '否'}")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        # INT8量化
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        if representative_dataset is not None and len(representative_dataset) > 0:
            # 使用代表性数据集进行全整数量化
            def representative_dataset_gen():
                for i in range(min(100, len(representative_dataset))):
                    yield [representative_dataset[i:i+1]]
            
            converter.representative_dataset = representative_dataset_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            print("使用全整数量化 (INT8)")
        else:
            print("使用动态范围量化")
    
    tflite_model = converter.convert()
    
    # 保存模型
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    model_size = os.path.getsize(output_path) / 1024
    print(f"✓ TFLite模型已保存到: {output_path}")
    print(f"  模型大小: {model_size:.2f} KB")
    
    return tflite_model


def evaluate_keras_model(model, X_test, y_test, class_names):
    """评估Keras模型"""
    print("\n" + "=" * 80)
    print("评估Keras原始模型")
    print("=" * 80)
    
    predictions = model.predict(X_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    
    # 计算准确率
    accuracy = np.mean(y_pred == y_test)
    print(f"\n总体准确率: {accuracy * 100:.2f}%")
    
    # 每类准确率
    print("\n各类别准确率:")
    print("-" * 60)
    for i, class_name in enumerate(class_names):
        mask = y_test == i
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred[mask] == y_test[mask])
            print(f"  {class_name:15s}: {class_acc * 100:5.2f}% ({np.sum(mask):3d} 样本)")
    
    # 混淆矩阵
    print("\n混淆矩阵:")
    print("-" * 60)
    confusion = np.zeros((len(class_names), len(class_names)), dtype=int)
    for true_label, pred_label in zip(y_test, y_pred):
        confusion[true_label, pred_label] += 1
    
    # 打印混淆矩阵
    print("        ", end="")
    for name in class_names:
        print(f"{name[:8]:>8s}", end=" ")
    print()
    
    for i, name in enumerate(class_names):
        print(f"{name[:8]:8s}", end=" ")
        for j in range(len(class_names)):
            print(f"{confusion[i, j]:8d}", end=" ")
        print()
    
    return accuracy, y_pred, predictions


def evaluate_tflite_model(tflite_model_path, X_test, y_test, class_names, is_quantized=False):
    """评估TFLite模型"""
    print("\n" + "=" * 80)
    print(f"评估TFLite模型 {'(量化)' if is_quantized else '(FLOAT32)'}")
    print("=" * 80)
    
    # 加载TFLite模型
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    # 获取输入输出细节
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"\n输入信息:")
    print(f"  形状: {input_details[0]['shape']}")
    print(f"  类型: {input_details[0]['dtype']}")
    
    print(f"\n输出信息:")
    print(f"  形状: {output_details[0]['shape']}")
    print(f"  类型: {output_details[0]['dtype']}")
    
    # 运行推理
    predictions = []
    y_pred = []
    
    print(f"\n运行推理 (共 {len(X_test)} 个样本)...")
    for i, sample in enumerate(X_test):
        if (i + 1) % 20 == 0:
            print(f"  进度: {i + 1}/{len(X_test)}")
        
        # 准备输入
        input_data = np.expand_dims(sample, axis=0)
        
        # 如果是量化模型,需要转换输入
        if is_quantized and input_details[0]['dtype'] == np.int8:
            input_scale, input_zero_point = input_details[0]['quantization']
            input_data = input_data / input_scale + input_zero_point
            input_data = input_data.astype(np.int8)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # 获取输出
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # 如果是量化模型,需要反量化输出
        if is_quantized and output_details[0]['dtype'] == np.int8:
            output_scale, output_zero_point = output_details[0]['quantization']
            output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
        
        predictions.append(output_data)
        y_pred.append(np.argmax(output_data))
    
    predictions = np.array(predictions)
    y_pred = np.array(y_pred)
    
    # 计算准确率
    accuracy = np.mean(y_pred == y_test)
    print(f"\n总体准确率: {accuracy * 100:.2f}%")
    
    # 每类准确率
    print("\n各类别准确率:")
    print("-" * 60)
    for i, class_name in enumerate(class_names):
        mask = y_test == i
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred[mask] == y_test[mask])
            print(f"  {class_name:15s}: {class_acc * 100:5.2f}% ({np.sum(mask):3d} 样本)")
    
    return accuracy, y_pred, predictions


def compare_predictions(keras_preds, tflite_preds, y_test, class_names):
    """比较Keras和TFLite模型的预测差异"""
    print("\n" + "=" * 80)
    print("模型预测比较")
    print("=" * 80)
    
    keras_pred_classes = np.argmax(keras_preds, axis=1)
    tflite_pred_classes = np.argmax(tflite_preds, axis=1)
    
    # 计算预测差异
    diff_mask = keras_pred_classes != tflite_pred_classes
    diff_count = np.sum(diff_mask)
    
    print(f"\n预测不同的样本数: {diff_count} / {len(y_test)} ({diff_count/len(y_test)*100:.2f}%)")
    
    if diff_count > 0:
        print("\n预测差异详情 (前10个):")
        print("-" * 80)
        count = 0
        for i in range(len(y_test)):
            if diff_mask[i] and count < 10:
                print(f"\n样本 {i}:")
                print(f"  真实类别: {class_names[y_test[i]]}")
                print(f"  Keras预测: {class_names[keras_pred_classes[i]]} (置信度: {keras_preds[i][keras_pred_classes[i]]*100:.2f}%)")
                print(f"  TFLite预测: {class_names[tflite_pred_classes[i]]} (置信度: {tflite_preds[i][tflite_pred_classes[i]]*100:.2f}%)")
                count += 1
    
    # 计算预测概率的平均绝对误差
    mae = np.mean(np.abs(keras_preds - tflite_preds))
    max_diff = np.max(np.abs(keras_preds - tflite_preds))
    
    print(f"\n预测概率差异:")
    print(f"  平均绝对误差 (MAE): {mae:.6f}")
    print(f"  最大绝对差异: {max_diff:.6f}")


def convert_to_c_array(tflite_model_path, output_path):
    """将TFLite模型转换为C数组"""
    print(f"\n转换为C数组...")
    
    with open(tflite_model_path, 'rb') as f:
        model_data = f.read()
    
    with open(output_path, 'w') as f:
        f.write('// Auto-generated file - DO NOT EDIT\n')
        f.write(f'// Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'// Source: {os.path.basename(tflite_model_path)}\n\n')
        f.write('#ifndef MODEL_DATA_H\n')
        f.write('#define MODEL_DATA_H\n\n')
        f.write('alignas(8) const unsigned char g_model[] = {\n')
        
        # 每行16个字节
        for i in range(0, len(model_data), 16):
            chunk = model_data[i:i+16]
            hex_str = ', '.join([f'0x{b:02x}' for b in chunk])
            f.write(f'  {hex_str},\n')
        
        f.write('};\n\n')
        f.write(f'const unsigned int g_model_len = {len(model_data)};\n\n')
        f.write('#endif  // MODEL_DATA_H\n')
    
    print(f"✓ C数组已保存到: {output_path}")


def main():
    base_path = Path(__file__).parent
    model_path = base_path / "model" / "CNN2D_ST_HandPosture_8classes.h5"
    dataset_path = base_path / "model" / "datasets" / "ST_VL53L8CX_handposture_dataset"
    
    class_names = [
        'None', 'FlatHand', 'Like', 'Dislike', 
        'Fist', 'Love', 'BreakTime', 'CrossHands'
    ]
    
    print("=" * 80)
    print("Keras to TFLite 转换与验证工具")
    print("=" * 80)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模型: {model_path}")
    
    # 1. 加载Keras模型
    print("\n" + "=" * 80)
    print("加载Keras模型")
    print("=" * 80)
    model = keras.models.load_model(str(model_path))
    print(f"✓ 模型加载成功")
    print(f"  输入形状: {model.input_shape}")
    print(f"  输出形状: {model.output_shape}")
    print(f"  总参数: {model.count_params():,}")
    
    # 2. 准备测试数据
    X_test, y_test = prepare_dataset(str(dataset_path), class_names, max_samples_per_class=None)
    
    # 3. 评估原始Keras模型
    keras_acc, keras_pred, keras_predictions = evaluate_keras_model(model, X_test, y_test, class_names)
    
    # 4. 转换为TFLite (FLOAT32)
    print("\n" + "=" * 80)
    print("转换为TFLite (FLOAT32)")
    print("=" * 80)
    tflite_float_path = base_path / "model_float32.tflite"
    convert_to_tflite(model, str(tflite_float_path), quantize=False)
    
    # 5. 评估TFLite FLOAT32模型
    tflite_float_acc, tflite_float_pred, tflite_float_predictions = evaluate_tflite_model(
        str(tflite_float_path), X_test, y_test, class_names, is_quantized=False
    )
    
    # 6. 比较FLOAT32预测
    compare_predictions(keras_predictions, tflite_float_predictions, y_test, class_names)
    
    # 7. 转换为TFLite (INT8量化)
    print("\n" + "=" * 80)
    print("转换为TFLite (INT8量化)")
    print("=" * 80)
    tflite_int8_path = base_path / "model_int8.tflite"
    convert_to_tflite(model, str(tflite_int8_path), quantize=True, representative_dataset=X_test)
    
    # 8. 评估TFLite INT8模型
    tflite_int8_acc, tflite_int8_pred, tflite_int8_predictions = evaluate_tflite_model(
        str(tflite_int8_path), X_test, y_test, class_names, is_quantized=True
    )
    
    # 9. 比较INT8预测
    compare_predictions(keras_predictions, tflite_int8_predictions, y_test, class_names)
    
    # 10. 生成C数组
    print("\n" + "=" * 80)
    print("生成C数组文件")
    print("=" * 80)
    
    c_array_float_path = base_path / "model_data_float32.h"
    convert_to_c_array(str(tflite_float_path), str(c_array_float_path))
    
    c_array_int8_path = base_path / "model_data_int8.h"
    convert_to_c_array(str(tflite_int8_path), str(c_array_int8_path))
    
    # 11. 总结
    print("\n" + "=" * 80)
    print("转换总结")
    print("=" * 80)
    
    print("\n模型文件:")
    print(f"  原始Keras模型: {os.path.getsize(model_path) / 1024:.2f} KB")
    print(f"  TFLite FLOAT32: {os.path.getsize(tflite_float_path) / 1024:.2f} KB")
    print(f"  TFLite INT8:    {os.path.getsize(tflite_int8_path) / 1024:.2f} KB")
    
    reduction = (1 - os.path.getsize(tflite_int8_path) / os.path.getsize(tflite_float_path)) * 100
    print(f"  量化减少:       {reduction:.1f}%")
    
    print("\n准确率对比:")
    print(f"  Keras原始模型:  {keras_acc * 100:.2f}%")
    print(f"  TFLite FLOAT32: {tflite_float_acc * 100:.2f}% (差异: {(tflite_float_acc - keras_acc) * 100:+.2f}%)")
    print(f"  TFLite INT8:    {tflite_int8_acc * 100:.2f}% (差异: {(tflite_int8_acc - keras_acc) * 100:+.2f}%)")
    
    print("\n生成的文件:")
    print(f"  ✓ {tflite_float_path}")
    print(f"  ✓ {tflite_int8_path}")
    print(f"  ✓ {c_array_float_path}")
    print(f"  ✓ {c_array_int8_path}")
    
    print("\n" + "=" * 80)
    print("转换完成!")
    print("=" * 80)
    
    print("\n下一步:")
    print("  1. 将 model_data_int8.h 复制到Arduino项目中")
    print("  2. 使用TensorFlow Lite Micro库进行部署")
    print("  3. 参考 MODEL_DATASET_ANALYSIS.md 中的代码示例")


if __name__ == "__main__":
    main()
