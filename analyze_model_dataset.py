#!/usr/bin/env python3
"""
分析ToF手势识别模型和数据集
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
from datetime import datetime


def analyze_keras_model(model_path):
    """分析Keras模型结构和参数"""
    print("=" * 80)
    print("模型分析")
    print("=" * 80)
    
    model = keras.models.load_model(model_path)
    
    print(f"\n模型路径: {model_path}")
    print(f"模型文件大小: {os.path.getsize(model_path) / 1024:.2f} KB\n")
    
    # 模型结构摘要
    print("模型结构:")
    print("-" * 80)
    model.summary()
    
    # 详细层信息
    print("\n\n详细层信息:")
    print("-" * 80)
    total_params = 0
    for i, layer in enumerate(model.layers):
        params = layer.count_params()
        total_params += params
        print(f"\n层 {i+1}: {layer.name}")
        print(f"  类型: {layer.__class__.__name__}")
        print(f"  输出形状: {layer.output_shape}")
        print(f"  参数数量: {params:,}")
        
        # 打印配置
        config = layer.get_config()
        important_configs = {}
        if 'activation' in config:
            important_configs['激活函数'] = config['activation']
        if 'filters' in config:
            important_configs['滤波器数量'] = config['filters']
        if 'kernel_size' in config:
            important_configs['卷积核大小'] = config['kernel_size']
        if 'strides' in config:
            important_configs['步长'] = config['strides']
        if 'padding' in config:
            important_configs['填充'] = config['padding']
        if 'pool_size' in config:
            important_configs['池化大小'] = config['pool_size']
        if 'rate' in config:
            important_configs['Dropout率'] = config['rate']
        if 'units' in config:
            important_configs['单元数'] = config['units']
            
        for key, value in important_configs.items():
            print(f"  {key}: {value}")
    
    print(f"\n总参数数量: {total_params:,}")
    
    # 输入输出信息
    print("\n\n输入/输出信息:")
    print("-" * 80)
    print(f"输入形状: {model.input_shape}")
    print(f"输出形状: {model.output_shape}")
    print(f"输入数据类型: {model.input.dtype}")
    print(f"输出数据类型: {model.output.dtype}")
    
    return model


def analyze_dataset(dataset_path, class_names):
    """分析数据集结构和统计信息"""
    print("\n\n" + "=" * 80)
    print("数据集分析")
    print("=" * 80)
    
    print(f"\n数据集路径: {dataset_path}")
    print(f"类别数量: {len(class_names)}")
    print(f"类别名称: {', '.join(class_names)}\n")
    
    dataset_stats = {}
    total_samples = 0
    
    for class_name in class_names:
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_path):
            print(f"警告: 类别 '{class_name}' 不存在")
            continue
            
        print(f"\n分析类别: {class_name}")
        print("-" * 80)
        
        # 查找所有子目录
        subdirs = [d for d in os.listdir(class_path) 
                  if os.path.isdir(os.path.join(class_path, d))]
        
        print(f"数据目录数量: {len(subdirs)}")
        
        class_samples = 0
        all_distances = []
        all_signals = []
        
        for subdir in subdirs:
            npz_path = os.path.join(class_path, subdir, 'npz')
            if not os.path.exists(npz_path):
                continue
                
            # 查找npz文件
            npz_files = [f for f in os.listdir(npz_path) 
                        if f.endswith('.npz') and not f.endswith('.npz:Zone.Identifier')]
            
            print(f"  {subdir}: {len(npz_files)} 个样本")
            
            # 加载几个样本进行分析
            sample_files = npz_files[:min(5, len(npz_files))]
            for npz_file in sample_files:
                try:
                    data = np.load(os.path.join(npz_path, npz_file), allow_pickle=True)
                    
                    # 检查数据结构
                    if class_samples == 0 and len(sample_files) > 0:
                        print(f"\n  样本文件结构 ({npz_file}):")
                        for key in data.files:
                            print(f"    - {key}: {data[key].shape if hasattr(data[key], 'shape') else type(data[key])}")
                    
                    # 提取zone_data
                    if 'zone_data' in data.files:
                        zone_data = data['zone_data'].item()
                        
                        if 'distance_mm' in zone_data:
                            distance = zone_data['distance_mm']
                            all_distances.extend(distance.flatten())
                            
                        if 'signal_per_spad' in zone_data:
                            signal = zone_data['signal_per_spad']
                            all_signals.extend(signal.flatten())
                            
                    class_samples += 1
                    
                except Exception as e:
                    print(f"  错误加载 {npz_file}: {e}")
            
            class_samples += len(npz_files) - len(sample_files)
        
        # 统计信息
        print(f"\n  总样本数: {class_samples}")
        if all_distances:
            all_distances = np.array(all_distances)
            print(f"  距离统计 (distance_mm):")
            print(f"    - 最小值: {np.min(all_distances):.2f} mm")
            print(f"    - 最大值: {np.max(all_distances):.2f} mm")
            print(f"    - 平均值: {np.mean(all_distances):.2f} mm")
            print(f"    - 标准差: {np.std(all_distances):.2f} mm")
        
        if all_signals:
            all_signals = np.array(all_signals)
            print(f"  信号统计 (signal_per_spad):")
            print(f"    - 最小值: {np.min(all_signals):.2f}")
            print(f"    - 最大值: {np.max(all_signals):.2f}")
            print(f"    - 平均值: {np.mean(all_signals):.2f}")
            print(f"    - 标准差: {np.std(all_signals):.2f}")
        
        dataset_stats[class_name] = class_samples
        total_samples += class_samples
    
    print("\n\n类别样本分布:")
    print("-" * 80)
    for class_name, count in dataset_stats.items():
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"{class_name:15s}: {count:5d} 样本 ({percentage:5.2f}%)")
    print(f"{'总计':15s}: {total_samples:5d} 样本")
    
    return dataset_stats


def analyze_data_sample(dataset_path, class_names):
    """详细分析单个数据样本"""
    print("\n\n" + "=" * 80)
    print("数据样本详细分析")
    print("=" * 80)
    
    for class_name in class_names[:3]:  # 只分析前3个类别
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_path):
            continue
            
        subdirs = [d for d in os.listdir(class_path) 
                  if os.path.isdir(os.path.join(class_path, d))]
        
        if not subdirs:
            continue
            
        npz_path = os.path.join(class_path, subdirs[0], 'npz')
        if not os.path.exists(npz_path):
            continue
            
        npz_files = [f for f in os.listdir(npz_path) 
                    if f.endswith('.npz') and not f.endswith('.npz:Zone.Identifier')]
        
        if not npz_files:
            continue
            
        # 加载第一个样本
        try:
            data = np.load(os.path.join(npz_path, npz_files[0]), allow_pickle=True)
            print(f"\n类别: {class_name}")
            print(f"文件: {npz_files[0]}")
            print("-" * 80)
            
            for key in data.files:
                value = data[key]
                print(f"\n键: {key}")
                print(f"类型: {type(value)}")
                
                if hasattr(value, 'shape'):
                    print(f"形状: {value.shape}")
                    print(f"数据类型: {value.dtype}")
                    
                if key == 'zone_data' and isinstance(value, np.ndarray) and value.size == 1:
                    zone_data = value.item()
                    if isinstance(zone_data, dict):
                        print(f"zone_data字典键: {list(zone_data.keys())}")
                        
                        if 'distance_mm' in zone_data:
                            dist = zone_data['distance_mm']
                            print(f"\ndistance_mm:")
                            print(f"  形状: {dist.shape}")
                            print(f"  数据类型: {dist.dtype}")
                            print(f"  数据预览:\n{dist}")
                            
                        if 'signal_per_spad' in zone_data:
                            signal = zone_data['signal_per_spad']
                            print(f"\nsignal_per_spad:")
                            print(f"  形状: {signal.shape}")
                            print(f"  数据类型: {signal.dtype}")
                            print(f"  数据预览:\n{signal}")
                            
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()


def main():
    # 配置路径
    base_path = Path(__file__).parent
    model_path = base_path / "model" / "CNN2D_ST_HandPosture_8classes.h5"
    dataset_path = base_path / "model" / "datasets" / "ST_VL53L8CX_handposture_dataset"
    
    class_names = [
        'None', 'FlatHand', 'Like', 'Dislike', 
        'Fist', 'Love', 'BreakTime', 'CrossHands'
    ]
    
    print("ToF手势识别模型和数据集分析报告")
    print("=" * 80)
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 分析模型
    if model_path.exists():
        model = analyze_keras_model(str(model_path))
    else:
        print(f"错误: 模型文件不存在: {model_path}")
        return
    
    # 分析数据集
    if dataset_path.exists():
        dataset_stats = analyze_dataset(str(dataset_path), class_names)
        analyze_data_sample(str(dataset_path), class_names)
    else:
        print(f"错误: 数据集路径不存在: {dataset_path}")
        return
    
    print("\n\n" + "=" * 80)
    print("Arduino移植要点")
    print("=" * 80)
    print("""
1. 输入数据格式:
   - 形状: (8, 8, 2)
   - 第一个通道: distance_mm (距离，单位毫米)
   - 第二个通道: signal_per_spad (每个SPAD的信号强度)
   - 数据类型: FLOAT32

2. 预处理要求:
   - 需要对输入数据进行归一化
   - 参考配置: Max_distance=400, Min_distance=100, Background_distance=120

3. 输出格式:
   - 8个类别的置信度分数
   - 使用softmax激活函数
   - 选择最高置信度的类别作为预测结果

4. 模型转换建议:
   - 使用TensorFlow Lite Micro进行转换
   - 或使用STM32Cube.AI进行优化和部署
   - 考虑量化以减少内存占用

5. 资源需求:
   - Flash: ~25 KB
   - RAM: ~3 KB
   - 推理时间: ~1.5 ms @ 84MHz (STM32F401)
""")


if __name__ == "__main__":
    main()
