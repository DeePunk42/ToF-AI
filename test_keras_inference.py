#!/usr/bin/env python3
"""
独立测试Keras模型推理
使用完整的STM32预处理流程
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import glob
from datetime import datetime


def load_and_preprocess_sample(npz_file, rotation=0):
    """
    加载并预处理单个样本 - 完全按照STM32代码的处理流程
    
    参数:
        npz_file: NPZ文件路径
        rotation: 旋转角度 (0, 90, 180, 270)
    """
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
        
        # 检查帧是否有效
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
        
        # 应用旋转
        if rotation == 90:
            input_tensor = np.rot90(input_tensor, k=1)  # 逆时针旋转90度
        elif rotation == 180:
            input_tensor = np.rot90(input_tensor, k=2)  # 旋转180度
        elif rotation == 270:
            input_tensor = np.rot90(input_tensor, k=3)  # 逆时针旋转270度 (顺时针90度)
        
        return input_tensor, is_valid_frame, min_distance
    except Exception as e:
        print(f"Error loading {npz_file}: {e}")
        return None, False, None


def test_single_sample(model, npz_file, class_names):
    """测试单个样本"""
    print("\n" + "=" * 80)
    print(f"测试样本: {os.path.basename(os.path.dirname(os.path.dirname(npz_file)))}")
    print("=" * 80)
    
    # 从路径中提取真实类别
    true_class_name = None
    for class_name in class_names:
        if class_name in npz_file:
            true_class_name = class_name
            break
    
    if true_class_name is None:
        print("⚠️  无法从路径提取类别名称")
        return
    
    true_class_idx = class_names.index(true_class_name)
    
    # 加载和预处理
    input_tensor, is_valid_frame, min_distance = load_and_preprocess_sample(npz_file)
    
    if input_tensor is None:
        print("❌ 加载失败")
        return
    
    print(f"\n文件: {os.path.basename(npz_file)}")
    print(f"真实类别: {true_class_name} (索引: {true_class_idx})")
    print(f"帧有效性: {'✓ 有效' if is_valid_frame else '✗ 无效'}")
    print(f"最小距离: {min_distance:.2f} mm")
    
    # 显示预处理后的数据统计
    print(f"\n预处理后的数据统计:")
    print(f"  距离通道: min={input_tensor[:,:,0].min():.3f}, max={input_tensor[:,:,0].max():.3f}, "
          f"mean={input_tensor[:,:,0].mean():.3f}, std={input_tensor[:,:,0].std():.3f}")
    print(f"  信号通道: min={input_tensor[:,:,1].min():.3f}, max={input_tensor[:,:,1].max():.3f}, "
          f"mean={input_tensor[:,:,1].mean():.3f}, std={input_tensor[:,:,1].std():.3f}")
    
    # 运行推理
    input_batch = np.expand_dims(input_tensor, axis=0)
    predictions = model.predict(input_batch, verbose=0)[0]
    
    # 显示所有类别的预测概率
    print(f"\n预测结果:")
    print("-" * 60)
    sorted_indices = np.argsort(predictions)[::-1]  # 从高到低排序
    
    for i, idx in enumerate(sorted_indices):
        marker = ""
        if idx == true_class_idx:
            marker = " ← 真实类别"
        elif i == 0:
            marker = " ← 预测类别"
        
        bar_length = int(predictions[idx] * 50)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"  {class_names[idx]:12s} {bar} {predictions[idx]*100:5.2f}%{marker}")
    
    # 判断预测结果
    predicted_class_idx = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_idx]
    confidence = predictions[predicted_class_idx] * 100
    
    print("\n" + "-" * 60)
    if predicted_class_idx == true_class_idx:
        print(f"✓ 预测正确: {predicted_class_name} (置信度: {confidence:.2f}%)")
    else:
        print(f"✗ 预测错误: 预测为 {predicted_class_name} (置信度: {confidence:.2f}%)")
        print(f"  真实类别: {true_class_name} (模型输出: {predictions[true_class_idx]*100:.2f}%)")


def test_batch_samples(model, dataset_path, class_names, samples_per_class=5, rotation=0):
    """批量测试多个样本"""
    print("\n" + "=" * 80)
    print(f"批量测试 (旋转: {rotation}°)")
    print("=" * 80)
    
    total_correct = 0
    total_samples = 0
    class_correct = {name: 0 for name in class_names}
    class_total = {name: 0 for name in class_names}
    
    for class_name in class_names:
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_dir):
            continue
        
        npz_files = glob.glob(os.path.join(class_dir, '**', 'npz', '*.npz'), recursive=True)
        npz_files = [f for f in npz_files if not f.endswith(':Zone.Identifier')]
        
        # 随机选择若干样本
        if len(npz_files) > samples_per_class:
            indices = np.random.choice(len(npz_files), samples_per_class, replace=False)
            npz_files = [npz_files[i] for i in indices]
        
        print(f"\n{class_name}: 测试 {len(npz_files)} 个样本")
        
        for npz_file in npz_files:
            input_tensor, is_valid_frame, _ = load_and_preprocess_sample(npz_file, rotation=rotation)
            
            if input_tensor is None:
                continue
            
            # 运行推理
            input_batch = np.expand_dims(input_tensor, axis=0)
            predictions = model.predict(input_batch, verbose=0)[0]
            predicted_class_idx = np.argmax(predictions)
            true_class_idx = class_names.index(class_name)
            
            total_samples += 1
            class_total[class_name] += 1
            
            if predicted_class_idx == true_class_idx:
                total_correct += 1
                class_correct[class_name] += 1
                print(f"  ✓ {os.path.basename(npz_file)}: {class_names[predicted_class_idx]} ({predictions[predicted_class_idx]*100:.1f}%)")
            else:
                print(f"  ✗ {os.path.basename(npz_file)}: 预测为 {class_names[predicted_class_idx]} ({predictions[predicted_class_idx]*100:.1f}%), 实际为 {class_name}")
    
    # 显示统计结果
    print("\n" + "=" * 80)
    print("测试结果统计")
    print("=" * 80)
    
    overall_accuracy = total_correct / total_samples * 100 if total_samples > 0 else 0
    print(f"\n总体准确率: {total_correct}/{total_samples} = {overall_accuracy:.2f}%")
    
    print("\n各类别准确率:")
    print("-" * 60)
    for class_name in class_names:
        if class_total[class_name] > 0:
            acc = class_correct[class_name] / class_total[class_name] * 100
            print(f"  {class_name:12s}: {class_correct[class_name]:2d}/{class_total[class_name]:2d} = {acc:5.2f}%")


def test_rotation_comparison(model, dataset_path, class_names, samples_per_class=5):
    """测试不同旋转角度对准确率的影响"""
    print("\n" + "=" * 80)
    print("旋转角度对比测试")
    print("=" * 80)
    
    rotations = [0, 90, 180, 270]
    results = {}
    
    # 设置随机种子以便复现
    np.random.seed(42)
    
    # 预加载所有测试样本
    test_samples = []
    for class_name in class_names:
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_dir):
            continue
        
        npz_files = glob.glob(os.path.join(class_dir, '**', 'npz', '*.npz'), recursive=True)
        npz_files = [f for f in npz_files if not f.endswith(':Zone.Identifier')]
        
        if len(npz_files) > samples_per_class:
            indices = np.random.choice(len(npz_files), samples_per_class, replace=False)
            npz_files = [npz_files[i] for i in indices]
        
        for npz_file in npz_files:
            test_samples.append((npz_file, class_name))
    
    print(f"\n总测试样本数: {len(test_samples)}")
    print(f"测试旋转角度: {rotations}")
    
    # 测试每个旋转角度
    for rotation in rotations:
        print(f"\n{'='*80}")
        print(f"测试旋转角度: {rotation}°")
        print(f"{'='*80}")
        
        correct = 0
        total = 0
        class_correct = {name: 0 for name in class_names}
        class_total = {name: 0 for name in class_names}
        
        for npz_file, true_class_name in test_samples:
            input_tensor, is_valid_frame, _ = load_and_preprocess_sample(npz_file, rotation=rotation)
            
            if input_tensor is None:
                continue
            
            # 运行推理
            input_batch = np.expand_dims(input_tensor, axis=0)
            predictions = model.predict(input_batch, verbose=0)[0]
            predicted_class_idx = np.argmax(predictions)
            true_class_idx = class_names.index(true_class_name)
            
            total += 1
            class_total[true_class_name] += 1
            
            if predicted_class_idx == true_class_idx:
                correct += 1
                class_correct[true_class_name] += 1
        
        accuracy = correct / total * 100 if total > 0 else 0
        results[rotation] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'class_correct': class_correct.copy(),
            'class_total': class_total.copy()
        }
        
        print(f"\n总体准确率: {correct}/{total} = {accuracy:.2f}%")
        
        # 显示各类别准确率
        print("\n各类别准确率:")
        print("-" * 60)
        for class_name in class_names:
            if class_total[class_name] > 0:
                acc = class_correct[class_name] / class_total[class_name] * 100
                print(f"  {class_name:12s}: {class_correct[class_name]:2d}/{class_total[class_name]:2d} = {acc:5.2f}%")
    
    # 显示对比总结
    print("\n" + "=" * 80)
    print("旋转角度对比总结")
    print("=" * 80)
    
    print(f"\n{'旋转角度':<12} {'准确率':<12} {'正确数':<12} {'总样本数':<12}")
    print("-" * 60)
    
    best_rotation = max(results.keys(), key=lambda r: results[r]['accuracy'])
    
    for rotation in rotations:
        r = results[rotation]
        marker = " ← 最佳" if rotation == best_rotation else ""
        print(f"{rotation:>3}°{' ':<7} {r['accuracy']:>6.2f}%{' ':<5} "
              f"{r['correct']:>4}/{r['total']:<4}{' ':<3} {r['total']:>4}{marker}")
    
    print(f"\n最佳旋转角度: {best_rotation}° (准确率: {results[best_rotation]['accuracy']:.2f}%)")
    
    # 显示每个类别在不同旋转下的最佳角度
    print("\n" + "=" * 80)
    print("各类别最佳旋转角度")
    print("=" * 80)
    print(f"\n{'类别':<12} {'最佳角度':<12} {'准确率':<12}")
    print("-" * 60)
    
    for class_name in class_names:
        best_class_rotation = None
        best_class_acc = -1
        
        for rotation in rotations:
            r = results[rotation]
            if r['class_total'][class_name] > 0:
                acc = r['class_correct'][class_name] / r['class_total'][class_name] * 100
                if acc > best_class_acc:
                    best_class_acc = acc
                    best_class_rotation = rotation
        
        if best_class_rotation is not None:
            print(f"{class_name:<12} {best_class_rotation:>3}°{' ':<7} {best_class_acc:>6.2f}%")
    
    return results


def visualize_input_data(input_tensor, title="输入数据可视化"):
    """可视化输入张量"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 距离通道
        im0 = axes[0].imshow(input_tensor[:, :, 0], cmap='viridis', aspect='auto')
        axes[0].set_title('距离通道 (归一化后)')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        plt.colorbar(im0, ax=axes[0])
        
        # 信号通道
        im1 = axes[1].imshow(input_tensor[:, :, 1], cmap='plasma', aspect='auto')
        axes[1].set_title('信号通道 (归一化后)')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[1])
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("⚠️  matplotlib未安装，跳过可视化")


def main():
    base_path = Path(__file__).parent
    model_path = base_path / "model" / "CNN2D_ST_HandPosture_8classes.h5"
    dataset_path = base_path / "model" / "datasets" / "ST_VL53L8CX_handposture_dataset"
    
    class_names = [
        'None', 'FlatHand', 'Like', 'Dislike', 
        'Fist', 'Love', 'BreakTime', 'CrossHands'
    ]
    
    print("=" * 80)
    print("Keras模型推理测试")
    print("=" * 80)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模型: {model_path}")
    print(f"数据集: {dataset_path}")
    
    # 加载模型
    print("\n加载Keras模型...")
    model = keras.models.load_model(str(model_path))
    print(f"✓ 模型加载成功")
    print(f"  输入形状: {model.input_shape}")
    print(f"  输出形状: {model.output_shape}")
    print(f"  总参数: {model.count_params():,}")
    
    # 模式选择
    print("\n" + "=" * 80)
    print("测试模式:")
    print("  1. 测试单个样本 (详细输出)")
    print("  2. 批量测试 (每类5个样本)")
    print("  3. 测试指定样本文件")
    print("  4. 旋转角度对比测试 (0°/90°/180°/270°)")
    print("=" * 80)
    
    mode = input("\n请选择模式 (1/2/3/4，默认2): ").strip() or "2"
    
    if mode == "1":
        # 测试单个样本
        print("\n可用类别:")
        for i, name in enumerate(class_names):
            print(f"  {i+1}. {name}")
        
        class_choice = input(f"\n请选择类别 (1-{len(class_names)}，默认2): ").strip() or "2"
        class_idx = int(class_choice) - 1
        class_name = class_names[class_idx]
        
        # 查找该类别的样本
        class_dir = dataset_path / class_name
        npz_files = list(class_dir.glob('**/npz/*.npz'))
        npz_files = [f for f in npz_files if not str(f).endswith(':Zone.Identifier')]
        
        if not npz_files:
            print(f"错误: 找不到类别 '{class_name}' 的样本")
            return
        
        # 随机选择一个样本
        npz_file = str(np.random.choice(npz_files))
        test_single_sample(model, npz_file, class_names)
        
        # 可视化输入数据
        if input("\n是否显示输入数据可视化? (y/n，默认n): ").strip().lower() == 'y':
            input_tensor, _, _ = load_and_preprocess_sample(npz_file)
            if input_tensor is not None:
                visualize_input_data(input_tensor, f"输入数据 - {class_name}")
    
    elif mode == "2":
        # 批量测试
        samples_per_class = input("\n每类测试样本数 (默认5): ").strip() or "5"
        samples_per_class = int(samples_per_class)
        
        rotation = input("旋转角度 (0/90/180/270，默认0): ").strip() or "0"
        rotation = int(rotation)
        
        # 设置随机种子以便复现
        np.random.seed(42)
        
        test_batch_samples(model, str(dataset_path), class_names, samples_per_class, rotation)
    
    elif mode == "3":
        # 测试指定文件
        npz_file = input("\n请输入NPZ文件完整路径: ").strip()
        
        if not os.path.exists(npz_file):
            print(f"错误: 文件不存在: {npz_file}")
            return
        
        test_single_sample(model, npz_file, class_names)
        
        # 可视化输入数据
        if input("\n是否显示输入数据可视化? (y/n，默认n): ").strip().lower() == 'y':
            input_tensor, _, _ = load_and_preprocess_sample(npz_file)
            if input_tensor is not None:
                visualize_input_data(input_tensor, "输入数据")
    
    elif mode == "4":
        # 旋转角度对比测试
        samples_per_class = input("\n每类测试样本数 (默认5): ").strip() or "5"
        samples_per_class = int(samples_per_class)
        
        test_rotation_comparison(model, str(dataset_path), class_names, samples_per_class)
    
    else:
        print("无效的模式选择")
        return
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
