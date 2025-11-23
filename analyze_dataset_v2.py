#!/usr/bin/env python3
"""
深入分析ToF手势识别数据集和模型
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from datetime import datetime


def load_sample_data(npz_file_path):
    """加载并解析单个npz样本文件"""
    data = np.load(npz_file_path, allow_pickle=True)
    
    zone_data = data['zone_data']  # shape: (4, 64, N)
    zone_head = data['zone_head']  # shape: (4,) - 字段名
    
    # 解析zone_head以找到signal_per_spad和distance_mm的索引
    zone_head_list = [str(h) for h in zone_head]
    
    # 通常顺序是: signal_per_spad, distance_mm, ambient_per_spad, nb_target_detected
    signal_idx = None
    distance_idx = None
    
    for i, name in enumerate(zone_head_list):
        if 'signal_per_spad' in name:
            signal_idx = i
        elif 'distance_mm' in name:
            distance_idx = i
    
    if signal_idx is None or distance_idx is None:
        return None, None, zone_head_list
    
    # zone_data shape: (4, 64, N)
    # 第一维: 不同的测量值类型
    # 第二维: 64个zone (8x8)
    # 第三维: 时间序列
    
    signal_data = zone_data[signal_idx, :, :]  # (64, N)
    distance_data = zone_data[distance_idx, :, :]  # (64, N)
    
    return signal_data, distance_data, zone_head_list


def analyze_model(model_path):
    """分析Keras模型"""
    print("=" * 80)
    print("模型结构分析")
    print("=" * 80)
    
    model = keras.models.load_model(model_path)
    
    print(f"\n模型文件: {model_path}")
    print(f"文件大小: {os.path.getsize(model_path) / 1024:.2f} KB\n")
    
    print("层结构详情:")
    print("-" * 80)
    
    for i, layer in enumerate(model.layers):
        print(f"\n第 {i+1} 层: {layer.name}")
        print(f"  类型: {layer.__class__.__name__}")
        print(f"  输出形状: {layer.output_shape}")
        print(f"  参数量: {layer.count_params():,}")
        
        config = layer.get_config()
        if 'activation' in config and config['activation']:
            print(f"  激活函数: {config['activation']}")
        if 'filters' in config:
            print(f"  滤波器: {config['filters']}")
        if 'kernel_size' in config:
            print(f"  卷积核: {config['kernel_size']}")
        if 'strides' in config and config['strides'] != (1, 1):
            print(f"  步长: {config['strides']}")
        if 'padding' in config:
            print(f"  填充: {config['padding']}")
        if 'pool_size' in config:
            print(f"  池化大小: {config['pool_size']}")
        if 'rate' in config:
            print(f"  Dropout率: {config['rate']}")
        if 'units' in config:
            print(f"  神经元数: {config['units']}")
    
    print(f"\n\n模型总参数: {model.count_params():,}")
    print(f"输入形状: {model.input_shape}")
    print(f"输出形状: {model.output_shape}")
    
    return model


def analyze_dataset_structure(dataset_path, class_names):
    """分析数据集结构"""
    print("\n\n" + "=" * 80)
    print("数据集结构分析")
    print("=" * 80)
    
    dataset_info = {}
    
    for class_name in class_names:
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_path):
            continue
        
        print(f"\n\n类别: {class_name}")
        print("-" * 80)
        
        subdirs = [d for d in os.listdir(class_path) 
                  if os.path.isdir(os.path.join(class_path, d))]
        
        total_files = 0
        sample_loaded = False
        
        for subdir in subdirs:
            npz_path = os.path.join(class_path, subdir, 'npz')
            if not os.path.exists(npz_path):
                continue
            
            npz_files = [f for f in os.listdir(npz_path) 
                        if f.endswith('.npz') and not f.endswith(':Zone.Identifier')]
            
            total_files += len(npz_files)
            
            # 加载第一个样本查看结构
            if not sample_loaded and npz_files:
                sample_file = os.path.join(npz_path, npz_files[0])
                signal, distance, headers = load_sample_data(sample_file)
                
                if signal is not None:
                    print(f"  数据字段: {headers}")
                    print(f"  Signal shape: {signal.shape}")
                    print(f"  Distance shape: {distance.shape}")
                    print(f"  时间序列长度: {signal.shape[1]}")
                    print(f"\n  Signal统计:")
                    print(f"    - 最小值: {np.min(signal):.2f}")
                    print(f"    - 最大值: {np.max(signal):.2f}")
                    print(f"    - 平均值: {np.mean(signal):.2f}")
                    print(f"\n  Distance统计 (mm):")
                    print(f"    - 最小值: {np.min(distance):.2f}")
                    print(f"    - 最大值: {np.max(distance):.2f}")
                    print(f"    - 平均值: {np.mean(distance):.2f}")
                    
                    # 显示一个8x8帧的示例
                    print(f"\n  单帧示例 (第1帧重塑为8x8):")
                    frame_distance = distance[:, 0].reshape(8, 8)
                    print("  Distance map:")
                    print(frame_distance.astype(int))
                    
                    frame_signal = signal[:, 0].reshape(8, 8)
                    print("\n  Signal map:")
                    print(frame_signal.astype(int))
                    
                    sample_loaded = True
        
        print(f"\n  总样本数: {total_files}")
        dataset_info[class_name] = total_files
    
    print("\n\n" + "=" * 80)
    print("数据集统计汇总")
    print("=" * 80)
    total = sum(dataset_info.values())
    for class_name, count in dataset_info.items():
        pct = (count / total * 100) if total > 0 else 0
        print(f"{class_name:15s}: {count:5d} 样本 ({pct:5.2f}%)")
    print(f"{'总计':15s}: {total:5d} 样本")
    
    return dataset_info


def main():
    base_path = Path(__file__).parent
    model_path = base_path / "model" / "CNN2D_ST_HandPosture_8classes.h5"
    dataset_path = base_path / "model" / "datasets" / "ST_VL53L8CX_handposture_dataset"
    
    class_names = [
        'None', 'FlatHand', 'Like', 'Dislike', 
        'Fist', 'Love', 'BreakTime', 'CrossHands'
    ]
    
    print("ToF手势识别系统 - 模型与数据集完整分析")
    print("=" * 80)
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"项目路径: {base_path}")
    
    # 分析模型
    if model_path.exists():
        model = analyze_model(str(model_path))
    else:
        print(f"错误: 模型文件不存在: {model_path}")
        return
    
    # 分析数据集
    if dataset_path.exists():
        dataset_info = analyze_dataset_structure(str(dataset_path), class_names)
    else:
        print(f"错误: 数据集路径不存在: {dataset_path}")
        return
    
    # Arduino移植要点
    print("\n\n" + "=" * 80)
    print("Arduino/STM32 移植要点")
    print("=" * 80)
    print("""
1. 数据输入规格:
   - 输入形状: (8, 8, 2)
   - 通道1: distance_mm - 距离测量值(毫米)
   - 通道2: signal_per_spad - 每个SPAD的信号强度
   - 数据类型: FLOAT32
   - 数据来源: VL53L8CX ToF传感器的64个zone

2. 数据预处理:
   - Distance归一化: (distance - 100) / (400 - 100)
   - Signal归一化: 需要根据实际数据范围确定
   - 背景距离阈值: 120mm
   - 数据格式转换: 从64个zone的1D数组重塑为8x8x2的3D张量

3. 模型规格:
   - 总参数: {:,}
   - Flash占用: ~25 KB (FLOAT32) 或 ~7 KB (INT8量化)
   - RAM占用: ~3 KB
   - 推理时间: ~1.5 ms @ 84MHz (STM32F401)

4. 模型转换流程:
   a. Keras (.h5) → TensorFlow Lite (.tflite)
   b. TFLite → TFLite Micro (C数组)
   c. 可选: 应用INT8量化减少模型大小

5. 代码实现要点:
   - 使用TensorFlow Lite Micro库
   - 分配tensor arena (约8KB)
   - 实现VL53L8CX驱动读取64-zone数据
   - 实现数据预处理函数
   - 调用推理引擎
   - 解析输出获取手势类别

6. 输出格式:
   - 8个类别的概率值 (0.0-1.0)
   - 使用softmax激活
   - 选择最大概率的类别作为识别结果
   - 类别: {}
""".format(model.count_params(), ', '.join(class_names)))


if __name__ == "__main__":
    main()
