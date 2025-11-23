#!/usr/bin/env python3
"""
可视化ToF数据和模型预测
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob


def load_sample_data(npz_file):
    """加载NPZ样本数据"""
    data = np.load(npz_file, allow_pickle=True)
    zone_data = data['zone_data']
    zone_head = data['zone_head']
    
    # 找到信号和距离的索引
    zone_head_list = [str(h) for h in zone_head]
    signal_idx = zone_head_list.index('signal_per_spad')
    distance_idx = zone_head_list.index('distance_mm')
    
    signal = zone_data[signal_idx, :, :]  # (64, N)
    distance = zone_data[distance_idx, :, :]  # (64, N)
    
    return signal, distance


def visualize_gesture(npz_file, frame_idx=0):
    """可视化单个手势的帧"""
    signal, distance = load_sample_data(npz_file)
    
    # 选择一帧
    if frame_idx >= signal.shape[1]:
        frame_idx = signal.shape[1] // 2
    
    signal_frame = signal[:, frame_idx].reshape(8, 8)
    distance_frame = distance[:, frame_idx].reshape(8, 8)
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 距离热力图
    im1 = axes[0].imshow(distance_frame, cmap='viridis', aspect='auto')
    axes[0].set_title('Distance Map (mm)')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0])
    
    # 在每个格子上显示数值
    for i in range(8):
        for j in range(8):
            text = axes[0].text(j, i, f'{int(distance_frame[i, j])}',
                              ha="center", va="center", color="w", fontsize=8)
    
    # 信号强度热力图
    im2 = axes[1].imshow(signal_frame, cmap='hot', aspect='auto')
    axes[1].set_title('Signal per SPAD')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[1])
    
    # 在每个格子上显示数值
    for i in range(8):
        for j in range(8):
            text = axes[1].text(j, i, f'{int(signal_frame[i, j])}',
                              ha="center", va="center", color="w", fontsize=8)
    
    # 文件名作为总标题
    gesture_name = Path(npz_file).parent.parent.name
    fig.suptitle(f'Gesture: {gesture_name} (Frame {frame_idx})', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def visualize_all_gestures(dataset_path, class_names):
    """可视化所有手势的示例"""
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    
    for idx, class_name in enumerate(class_names):
        class_path = Path(dataset_path) / class_name
        if not class_path.exists():
            continue
        
        # 找到第一个NPZ文件
        npz_files = list(glob.glob(str(class_path / '**' / 'npz' / '*.npz'), recursive=True))
        npz_files = [f for f in npz_files if not f.endswith(':Zone.Identifier')]
        
        if not npz_files:
            continue
        
        # 加载数据
        signal, distance = load_sample_data(npz_files[0])
        frame_idx = signal.shape[1] // 2
        distance_frame = distance[:, frame_idx].reshape(8, 8)
        
        # 绘制距离图
        im = axes[idx].imshow(distance_frame, cmap='viridis', aspect='auto')
        axes[idx].set_title(class_name, fontsize=12, fontweight='bold')
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx], fraction=0.046)
    
    # 隐藏多余的子图
    for idx in range(len(class_names), 16):
        axes[idx].axis('off')
    
    fig.suptitle('All Gesture Classes - Distance Maps', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_time_series(npz_file):
    """绘制时间序列数据"""
    signal, distance = load_sample_data(npz_file)
    
    # 计算平均值(所有zone)
    avg_signal = np.mean(signal, axis=0)
    avg_distance = np.mean(distance, axis=0)
    
    # 创建图形
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 距离时间序列
    axes[0].plot(avg_distance, 'b-', linewidth=2)
    axes[0].set_title('Average Distance over Time')
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Distance (mm)')
    axes[0].grid(True, alpha=0.3)
    
    # 信号时间序列
    axes[1].plot(avg_signal, 'r-', linewidth=2)
    axes[1].set_title('Average Signal per SPAD over Time')
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Signal Intensity')
    axes[1].grid(True, alpha=0.3)
    
    gesture_name = Path(npz_file).parent.parent.name
    fig.suptitle(f'Time Series Analysis - {gesture_name}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def main():
    base_path = Path(__file__).parent
    dataset_path = base_path / "model" / "datasets" / "ST_VL53L8CX_handposture_dataset"
    
    class_names = [
        'None', 'FlatHand', 'Like', 'Dislike', 
        'Fist', 'Love', 'BreakTime', 'CrossHands'
    ]
    
    print("ToF数据可视化")
    print("=" * 60)
    
    # 1. 可视化所有手势类别
    print("\n生成所有手势类别概览...")
    try:
        fig1 = visualize_all_gestures(str(dataset_path), class_names)
        output_file = base_path / "all_gestures_overview.png"
        fig1.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ 保存到: {output_file}")
    except Exception as e:
        print(f"✗ 错误: {e}")
    
    # 2. 为每个手势类别生成详细视图
    print("\n生成各手势详细视图...")
    for class_name in class_names[:4]:  # 只生成前4个示例
        class_path = dataset_path / class_name
        if not class_path.exists():
            continue
        
        npz_files = list(glob.glob(str(class_path / '**' / 'npz' / '*.npz'), recursive=True))
        npz_files = [f for f in npz_files if not f.endswith(':Zone.Identifier')]
        
        if npz_files:
            try:
                # 单帧可视化
                fig2 = visualize_gesture(npz_files[0])
                output_file = base_path / f"gesture_{class_name}_frame.png"
                fig2.savefig(output_file, dpi=150, bbox_inches='tight')
                print(f"✓ {class_name}: {output_file}")
                
                # 时间序列
                fig3 = plot_time_series(npz_files[0])
                output_file = base_path / f"gesture_{class_name}_timeseries.png"
                fig3.savefig(output_file, dpi=150, bbox_inches='tight')
                print(f"✓ {class_name} 时间序列: {output_file}")
                
                plt.close('all')
            except Exception as e:
                print(f"✗ {class_name} 错误: {e}")
    
    print("\n" + "=" * 60)
    print("可视化完成!")
    print("\n生成的文件:")
    print("  - all_gestures_overview.png")
    print("  - gesture_*_frame.png")
    print("  - gesture_*_timeseries.png")


if __name__ == "__main__":
    main()
