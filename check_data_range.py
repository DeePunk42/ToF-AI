#!/usr/bin/env python3
"""
检查数据集中实际的距离和信号范围
"""
import os
import numpy as np
import glob
from pathlib import Path


def analyze_data_range(dataset_path, class_names):
    """分析数据集中的实际数值范围"""
    all_distances = []
    all_signals = []
    
    print("分析数据集中的实际数值范围...")
    print("-" * 80)
    
    for class_name in class_names:
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_dir):
            continue
        
        npz_files = glob.glob(os.path.join(class_dir, '**', 'npz', '*.npz'), recursive=True)
        npz_files = [f for f in npz_files if not f.endswith(':Zone.Identifier')]
        
        print(f"\n{class_name}: 分析 {len(npz_files)} 个样本")
        
        for npz_file in npz_files:
            try:
                data = np.load(npz_file, allow_pickle=True)
                zone_data = data['zone_data']  # (4, 64, N)
                zone_head = data['zone_head']
                
                # 找到distance_mm和signal_per_spad的索引
                zone_head_list = [str(h) for h in zone_head]
                signal_idx = zone_head_list.index('signal_per_spad')
                distance_idx = zone_head_list.index('distance_mm')
                
                # 提取所有帧的数据
                signal = zone_data[signal_idx, :, :]  # (64, N)
                distance = zone_data[distance_idx, :, :]  # (64, N)
                
                # 收集所有值
                all_distances.extend(distance.flatten())
                all_signals.extend(signal.flatten())
                
            except Exception as e:
                print(f"  警告: 加载 {npz_file} 失败: {e}")
    
    # 转换为numpy数组
    all_distances = np.array(all_distances)
    all_signals = np.array(all_signals)
    
    # 计算统计信息
    print("\n" + "=" * 80)
    print("统计信息")
    print("=" * 80)
    
    print("\n距离数据 (distance_mm):")
    print(f"  最小值: {np.min(all_distances):.2f}")
    print(f"  最大值: {np.max(all_distances):.2f}")
    print(f"  平均值: {np.mean(all_distances):.2f}")
    print(f"  中位数: {np.median(all_distances):.2f}")
    print(f"  标准差: {np.std(all_distances):.2f}")
    print(f"  25%分位: {np.percentile(all_distances, 25):.2f}")
    print(f"  75%分位: {np.percentile(all_distances, 75):.2f}")
    print(f"  IQR (四分位距): {np.percentile(all_distances, 75) - np.percentile(all_distances, 25):.2f}")
    
    print("\n信号数据 (signal_per_spad):")
    print(f"  最小值: {np.min(all_signals):.2f}")
    print(f"  最大值: {np.max(all_signals):.2f}")
    print(f"  平均值: {np.mean(all_signals):.2f}")
    print(f"  中位数: {np.median(all_signals):.2f}")
    print(f"  标准差: {np.std(all_signals):.2f}")
    print(f"  25%分位: {np.percentile(all_signals, 25):.2f}")
    print(f"  75%分位: {np.percentile(all_signals, 75):.2f}")
    print(f"  IQR (四分位距): {np.percentile(all_signals, 75) - np.percentile(all_signals, 25):.2f}")
    
    # 比较与代码中的归一化参数
    print("\n" + "=" * 80)
    print("与代码中的归一化参数对比")
    print("=" * 80)
    
    print("\n距离归一化参数:")
    print(f"  代码中的中位数: 295")
    print(f"  实际中位数: {np.median(all_distances):.2f}")
    print(f"  代码中的IQR: 196")
    print(f"  实际IQR: {np.percentile(all_distances, 75) - np.percentile(all_distances, 25):.2f}")
    
    print("\n信号归一化参数:")
    print(f"  代码中的中位数: 281")
    print(f"  实际中位数: {np.median(all_signals):.2f}")
    print(f"  代码中的IQR: 452")
    print(f"  实际IQR: {np.percentile(all_signals, 75) - np.percentile(all_signals, 25):.2f}")
    
    # 检查数据分布
    print("\n" + "=" * 80)
    print("数据分布检查")
    print("=" * 80)
    
    print("\n距离分布:")
    ranges = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500), (500, 1000), (1000, 5000)]
    for r_min, r_max in ranges:
        count = np.sum((all_distances >= r_min) & (all_distances < r_max))
        percentage = count / len(all_distances) * 100
        print(f"  {r_min:4d}-{r_max:4d} mm: {count:7d} ({percentage:5.2f}%)")
    
    print("\n信号分布:")
    ranges = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500), (500, 1000), (1000, 5000)]
    for r_min, r_max in ranges:
        count = np.sum((all_signals >= r_min) & (all_signals < r_max))
        percentage = count / len(all_signals) * 100
        print(f"  {r_min:4d}-{r_max:4d}:     {count:7d} ({percentage:5.2f}%)")


def main():
    base_path = Path(__file__).parent
    dataset_path = base_path / "model" / "datasets" / "ST_VL53L8CX_handposture_dataset"
    
    class_names = [
        'None', 'FlatHand', 'Like', 'Dislike', 
        'Fist', 'Love', 'BreakTime', 'CrossHands'
    ]
    
    analyze_data_range(str(dataset_path), class_names)


if __name__ == "__main__":
    main()
