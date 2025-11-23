#!/usr/bin/env python3
"""
测试模型输入输出,找出正确的预处理方法
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import glob


def test_model_with_raw_data():
    """使用原始数据测试模型"""
    base_path = Path(__file__).parent
    model_path = base_path / "model" / "CNN2D_ST_HandPosture_8classes.h5"
    dataset_path = base_path / "model" / "datasets" / "ST_VL53L8CX_handposture_dataset"
    
    # 加载模型
    print("加载模型...")
    model = keras.models.load_model(str(model_path))
    print(f"输入形状: {model.input_shape}")
    print(f"输出形状: {model.output_shape}")
    
    # 加载一个样本
    class_name = "FlatHand"
    class_path = dataset_path / class_name
    npz_files = list(glob.glob(str(class_path / '**' / 'npz' / '*.npz'), recursive=True))
    npz_files = [f for f in npz_files if not f.endswith(':Zone.Identifier')]
    
    print(f"\n测试样本: {npz_files[0]}")
    
    # 加载数据
    data = np.load(npz_files[0], allow_pickle=True)
    zone_data = data['zone_data']
    zone_head = data['zone_head']
    
    print(f"Zone data shape: {zone_data.shape}")
    print(f"Zone headers: {zone_head}")
    
    zone_head_list = [str(h) for h in zone_head]
    signal_idx = zone_head_list.index('signal_per_spad')
    distance_idx = zone_head_list.index('distance_mm')
    
    signal = zone_data[signal_idx, :, :]  # (64, N)
    distance = zone_data[distance_idx, :, :]  # (64, N)
    
    print(f"\nSignal shape: {signal.shape}")
    print(f"Distance shape: {distance.shape}")
    
    # 尝试不同的预处理方法
    frame_idx = signal.shape[1] // 2
    
    print(f"\n使用帧 {frame_idx}/{signal.shape[1]}")
    
    signal_frame = signal[:, frame_idx]
    distance_frame = distance[:, frame_idx]
    
    print(f"\nDistance统计:")
    print(f"  Min: {np.min(distance_frame):.2f}")
    print(f"  Max: {np.max(distance_frame):.2f}")
    print(f"  Mean: {np.mean(distance_frame):.2f}")
    
    print(f"\nSignal统计:")
    print(f"  Min: {np.min(signal_frame):.2f}")
    print(f"  Max: {np.max(signal_frame):.2f}")
    print(f"  Mean: {np.mean(signal_frame):.2f}")
    
    # 测试不同的归一化方法
    methods = [
        ("无归一化", distance_frame, signal_frame),
        ("Distance/1000, Signal/1000", distance_frame/1000, signal_frame/1000),
        ("Distance clip[100,400]归一化, Signal/5000", 
         (np.clip(distance_frame, 100, 400) - 100) / 300, 
         signal_frame / 5000),
        ("原始值除以均值", 
         distance_frame / np.mean(distance_frame), 
         signal_frame / np.mean(signal_frame)),
    ]
    
    print("\n" + "=" * 80)
    print("测试不同的预处理方法")
    print("=" * 80)
    
    for method_name, dist_proc, sig_proc in methods:
        # 构建输入
        input_tensor = np.zeros((1, 8, 8, 2), dtype=np.float32)
        input_tensor[0, :, :, 0] = dist_proc.reshape(8, 8)
        input_tensor[0, :, :, 1] = sig_proc.reshape(8, 8)
        
        # 预测
        pred = model.predict(input_tensor, verbose=0)[0]
        pred_class = np.argmax(pred)
        
        class_names = ['None', 'FlatHand', 'Like', 'Dislike', 'Fist', 'Love', 'BreakTime', 'CrossHands']
        
        print(f"\n方法: {method_name}")
        print(f"  预测: {class_names[pred_class]} (置信度: {pred[pred_class]*100:.2f}%)")
        print(f"  所有概率: {[f'{p*100:.1f}%' for p in pred]}")
        
        # 检查输入数值范围
        print(f"  输入范围: Distance [{np.min(dist_proc):.3f}, {np.max(dist_proc):.3f}], "
              f"Signal [{np.min(sig_proc):.3f}, {np.max(sig_proc):.3f}]")


if __name__ == "__main__":
    test_model_with_raw_data()
