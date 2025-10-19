"""
TFLite模型验证脚本
验证量化后的TFLite模型在各个类别数据集上的准确性
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path
import glob
from collections import defaultdict
import json


class TFLiteModelValidator:
    """TFLite模型验证器"""

    def __init__(self, tflite_model_path, dataset_path):
        """
        初始化验证器

        Args:
            tflite_model_path: TFLite模型路径
            dataset_path: 数据集根目录路径
        """
        self.tflite_model_path = tflite_model_path
        self.dataset_path = dataset_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None

        # 类别定义
        self.class_names = [
            "None", "Like", "Dislike", "FlatHand",
            "Fist", "Love", "BreakTime", "CrossHands"
        ]

        # 加载模型
        self._load_model()

    def _load_model(self):
        """加载TFLite模型"""
        print(f"加载TFLite模型: {self.tflite_model_path}")
        self.interpreter = tf.lite.Interpreter(model_path=self.tflite_model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

        print("\n模型信息:")
        print(f"  输入形状: {self.input_details['shape']}")
        print(f"  输入类型: {self.input_details['dtype']}")
        print(f"  输出形状: {self.output_details['shape']}")
        print(f"  输出类型: {self.output_details['dtype']}")

        # 获取量化参数
        if self.input_details['dtype'] == np.int8:
            input_quant = self.input_details['quantization_parameters']
            output_quant = self.output_details['quantization_parameters']

            self.input_scale = input_quant['scales'][0] if input_quant['scales'].size > 0 else 1.0
            self.input_zero_point = input_quant['zero_points'][0] if input_quant['zero_points'].size > 0 else 0

            self.output_scale = output_quant['scales'][0] if output_quant['scales'].size > 0 else 1.0
            self.output_zero_point = output_quant['zero_points'][0] if output_quant['zero_points'].size > 0 else 0

            print(f"\n量化参数:")
            print(f"  输入 - Scale: {self.input_scale}, Zero point: {self.input_zero_point}")
            print(f"  输出 - Scale: {self.output_scale}, Zero point: {self.output_zero_point}")
        else:
            self.input_scale = 1.0
            self.input_zero_point = 0
            self.output_scale = 1.0
            self.output_zero_point = 0

    def load_dataset(self, max_samples_per_class=None):
        """
        加载数据集

        Args:
            max_samples_per_class: 每个类别最大样本数，None表示加载全部

        Returns:
            tuple: (samples, labels) - 样本数组和对应标签
        """
        print(f"\n从 {self.dataset_path} 加载数据集...")

        samples = []
        labels = []
        class_sample_counts = defaultdict(int)

        for class_idx, class_name in enumerate(self.class_names):
            class_dir = Path(self.dataset_path) / class_name

            if not class_dir.exists():
                print(f"警告: 类别目录不存在: {class_dir}")
                continue

            # 查找所有NPZ文件
            npz_pattern = str(class_dir / "**" / "*.npz")
            npz_files = glob.glob(npz_pattern, recursive=True)

            # 过滤掉Zone.Identifier文件
            npz_files = [f for f in npz_files if not f.endswith(':Zone.Identifier')]

            # 限制样本数量
            if max_samples_per_class is not None:
                npz_files = npz_files[:max_samples_per_class]

            # 加载每个NPZ文件
            for npz_file in npz_files:
                try:
                    data = self._load_npz_file(npz_file)
                    if data is not None:
                        samples.append(data)
                        labels.append(class_idx)
                        class_sample_counts[class_name] += 1
                except Exception as e:
                    print(f"  警告: 无法加载 {npz_file}: {e}")
                    continue

        print(f"\n数据集加载完成:")
        print(f"  总样本数: {len(samples)}")
        print(f"  各类别样本数:")
        for class_name in self.class_names:
            count = class_sample_counts[class_name]
            print(f"    {class_name:12s}: {count:4d} 样本")

        return np.array(samples), np.array(labels)

    def _load_npz_file(self, npz_file):
        """
        加载单个NPZ文件

        Args:
            npz_file: NPZ文件路径

        Returns:
            np.ndarray: (8, 8, 2)形状的数据，或None如果加载失败
        """
        data = np.load(npz_file)

        # 方法1: ST VL53L8CX格式 - zone_data字段
        if 'zone_data' in data and 'zone_head' in data:
            try:
                zone_data = data['zone_data']  # 形状: (frames, 64, features)
                zone_head = data['zone_head']  # 列名: ['target_status', 'valid', 'signal_per_spad', 'distance_mm']

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

                # 取每行的第一个值（或平均值）
                distance = distance_cols[:, 0] if distance_cols.shape[1] > 0 else distance_cols.flatten()
                signal = signal_cols[:, 0] if signal_cols.shape[1] > 0 else signal_cols.flatten()

                # Reshape到8x8
                distance_8x8 = distance.reshape(8, 8)
                signal_8x8 = signal.reshape(8, 8)

                # 合并为(8, 8, 2)
                combined = np.stack([distance_8x8, signal_8x8], axis=-1)
                return combined.astype(np.float32)

            except Exception as e:
                print(f"  警告: 解析zone_data失败: {e}")
                return None

        # 方法2: 分离的distance和signal字段
        elif 'distance_mm' in data and 'signal_per_spad' in data:
            distance = data['distance_mm']
            signal = data['signal_per_spad']

            # 确保形状正确
            if distance.shape == (8, 8) and signal.shape == (8, 8):
                combined = np.stack([distance, signal], axis=-1)
                return combined.astype(np.float32)
            else:
                # 可能需要reshape
                distance = np.array(distance).reshape(8, 8)
                signal = np.array(signal).reshape(8, 8)
                combined = np.stack([distance, signal], axis=-1)
                return combined.astype(np.float32)

        # 方法3: 单一数组
        elif 'arr_0' in data:
            arr = data['arr_0']
            if arr.shape == (8, 8, 2):
                return arr.astype(np.float32)
            elif arr.shape == (128,):  # 8*8*2 = 128
                return arr.reshape(8, 8, 2).astype(np.float32)

        # 方法4: 'data'字段
        elif 'data' in data:
            arr = data['data']
            if arr.shape == (8, 8, 2):
                return arr.astype(np.float32)
            elif arr.shape == (128,):
                return arr.reshape(8, 8, 2).astype(np.float32)

        # 尝试自动检测
        for key in data.files:
            arr = data[key]
            if isinstance(arr, np.ndarray):
                if arr.shape == (8, 8, 2):
                    return arr.astype(np.float32)
                elif arr.shape == (128,):
                    return arr.reshape(8, 8, 2).astype(np.float32)

        print(f"  警告: 无法解析NPZ文件结构: {npz_file}")
        print(f"    可用字段: {data.files}")
        return None

    def preprocess_input(self, data):
        """
        预处理输入数据

        Args:
            data: (8, 8, 2)形状的float32数据

        Returns:
            预处理后的数据，匹配模型输入类型
        """
        # 根据配置文件应用预处理
        MAX_DISTANCE = 400
        MIN_DISTANCE = 100
        BACKGROUND_DISTANCE = 120

        # 处理距离通道
        distance = data[:, :, 0].copy()
        distance = np.clip(distance, MIN_DISTANCE, MAX_DISTANCE)
        distance[distance < MIN_DISTANCE] = BACKGROUND_DISTANCE

        # 归一化（可选，根据训练时的预处理调整）
        # distance = (distance - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)

        # 重新组合
        processed = data.copy()
        processed[:, :, 0] = distance

        # 添加batch维度
        processed = np.expand_dims(processed, axis=0)

        # 量化到INT8（如果需要）
        if self.input_details['dtype'] == np.int8:
            processed = processed / self.input_scale + self.input_zero_point
            processed = np.clip(processed, -128, 127).astype(np.int8)
        else:
            processed = processed.astype(np.float32)

        return processed

    def predict(self, data):
        """
        对单个样本进行推理

        Args:
            data: (8, 8, 2)形状的输入数据

        Returns:
            tuple: (predicted_class, probabilities)
        """
        # 预处理
        input_data = self.preprocess_input(data)

        # 设置输入
        self.interpreter.set_tensor(self.input_details['index'], input_data)

        # 运行推理
        self.interpreter.invoke()

        # 获取输出
        output_data = self.interpreter.get_tensor(self.output_details['index'])[0]

        # 反量化（如果是INT8）
        if self.output_details['dtype'] == np.int8:
            output_data = (output_data.astype(np.float32) - self.output_zero_point) * self.output_scale

        # Softmax（如果输出不是概率）
        probabilities = self._softmax(output_data)

        # 获取预测类别
        predicted_class = np.argmax(probabilities)

        return predicted_class, probabilities

    def _softmax(self, x):
        """Softmax函数"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def evaluate(self, samples, labels):
        """
        评估模型性能

        Args:
            samples: 样本数组
            labels: 真实标签

        Returns:
            dict: 评估结果
        """
        print(f"\n开始评估模型 (共 {len(samples)} 个样本)...")

        predictions = []
        confidences = []

        # 逐个样本推理
        for i, sample in enumerate(samples):
            if (i + 1) % 100 == 0:
                print(f"  进度: {i + 1}/{len(samples)}")

            pred_class, probs = self.predict(sample)
            predictions.append(pred_class)
            confidences.append(probs[pred_class])

        predictions = np.array(predictions)
        confidences = np.array(confidences)

        # 计算准确率
        accuracy = np.mean(predictions == labels)

        # 计算混淆矩阵
        confusion_matrix = self._compute_confusion_matrix(labels, predictions, len(self.class_names))

        # 计算每个类别的指标
        class_metrics = self._compute_class_metrics(labels, predictions, confusion_matrix)

        results = {
            'accuracy': accuracy,
            'confusion_matrix': confusion_matrix,
            'class_metrics': class_metrics,
            'predictions': predictions,
            'confidences': confidences,
            'labels': labels
        }

        return results

    def _compute_confusion_matrix(self, y_true, y_pred, num_classes):
        """计算混淆矩阵"""
        cm = np.zeros((num_classes, num_classes), dtype=np.int32)
        for true, pred in zip(y_true, y_pred):
            cm[true][pred] += 1
        return cm

    def _compute_class_metrics(self, y_true, y_pred, confusion_matrix):
        """计算每个类别的精确率、召回率、F1分数"""
        metrics = {}

        for i, class_name in enumerate(self.class_names):
            tp = confusion_matrix[i][i]
            fp = confusion_matrix[:, i].sum() - tp
            fn = confusion_matrix[i, :].sum() - tp
            tn = confusion_matrix.sum() - tp - fp - fn

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': int(confusion_matrix[i, :].sum())
            }

        return metrics

    def print_results(self, results):
        """打印评估结果"""
        print("\n" + "="*70)
        print("评估结果")
        print("="*70)

        print(f"\n总体准确率: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")

        print(f"\n平均置信度: {results['confidences'].mean():.4f}")

        # 打印混淆矩阵
        print("\n混淆矩阵:")
        print("真实\\预测", end="")
        for class_name in self.class_names:
            print(f"{class_name:>12s}", end="")
        print()

        cm = results['confusion_matrix']
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name:10s}", end="")
            for j in range(len(self.class_names)):
                print(f"{cm[i][j]:12d}", end="")
            print()

        # 打印每个类别的指标
        print("\n" + "-"*70)
        print("各类别详细指标:")
        print("-"*70)
        print(f"{'类别':<12s} {'精确率':>10s} {'召回率':>10s} {'F1分数':>10s} {'样本数':>10s}")
        print("-"*70)

        class_metrics = results['class_metrics']
        for class_name in self.class_names:
            metrics = class_metrics[class_name]
            print(f"{class_name:<12s} "
                  f"{metrics['precision']:>10.4f} "
                  f"{metrics['recall']:>10.4f} "
                  f"{metrics['f1_score']:>10.4f} "
                  f"{metrics['support']:>10d}")

        # 计算加权平均
        total_support = sum(m['support'] for m in class_metrics.values())
        if total_support > 0:
            weighted_precision = sum(m['precision'] * m['support'] for m in class_metrics.values()) / total_support
            weighted_recall = sum(m['recall'] * m['support'] for m in class_metrics.values()) / total_support
            weighted_f1 = sum(m['f1_score'] * m['support'] for m in class_metrics.values()) / total_support

            print("-"*70)
            print(f"{'加权平均':<12s} "
                  f"{weighted_precision:>10.4f} "
                  f"{weighted_recall:>10.4f} "
                  f"{weighted_f1:>10.4f} "
                  f"{total_support:>10d}")

        print("="*70)

    def save_results(self, results, output_path):
        """保存评估结果到JSON文件"""
        # 转换numpy类型为Python原生类型
        results_serializable = {
            'accuracy': float(results['accuracy']),
            'mean_confidence': float(results['confidences'].mean()),
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'class_metrics': results['class_metrics'],
            'model_path': self.tflite_model_path,
            'dataset_path': self.dataset_path,
            'num_samples': len(results['labels'])
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)

        print(f"\n结果已保存到: {output_path}")

    def analyze_errors(self, results, num_examples=5):
        """分析错误预测的样本"""
        print("\n" + "="*70)
        print("错误样本分析")
        print("="*70)

        predictions = results['predictions']
        labels = results['labels']
        confidences = results['confidences']

        # 找出所有错误预测
        errors = np.where(predictions != labels)[0]

        print(f"\n错误预测数量: {len(errors)} / {len(labels)} "
              f"({len(errors)/len(labels)*100:.2f}%)")

        if len(errors) == 0:
            print("没有错误预测！")
            return

        # 显示置信度最高的错误样本
        error_confidences = confidences[errors]
        sorted_indices = np.argsort(error_confidences)[::-1]  # 降序

        print(f"\n置信度最高的 {min(num_examples, len(errors))} 个错误样本:")
        print("-"*70)
        print(f"{'编号':<6s} {'真实类别':<12s} {'预测类别':<12s} {'置信度':>10s}")
        print("-"*70)

        for i in range(min(num_examples, len(errors))):
            idx = errors[sorted_indices[i]]
            true_class = self.class_names[labels[idx]]
            pred_class = self.class_names[predictions[idx]]
            confidence = confidences[idx]

            print(f"{idx:<6d} {true_class:<12s} {pred_class:<12s} {confidence:>10.4f}")

        print("="*70)


def compare_models(model_paths, dataset_path, max_samples_per_class=50):
    """
    比较多个模型的性能

    Args:
        model_paths: 模型路径列表
        dataset_path: 数据集路径
        max_samples_per_class: 每个类别最大样本数
    """
    print("="*70)
    print("模型对比评估")
    print("="*70)

    results_all = {}

    for model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"\n警告: 模型文件不存在: {model_path}")
            continue

        print(f"\n{'='*70}")
        print(f"评估模型: {os.path.basename(model_path)}")
        print(f"{'='*70}")

        validator = TFLiteModelValidator(model_path, dataset_path)
        samples, labels = validator.load_dataset(max_samples_per_class=max_samples_per_class)
        results = validator.evaluate(samples, labels)
        validator.print_results(results)

        model_name = os.path.basename(model_path)
        results_all[model_name] = results['accuracy']

    # 打印对比摘要
    print("\n" + "="*70)
    print("模型对比摘要")
    print("="*70)
    print(f"{'模型名称':<40s} {'准确率':>15s}")
    print("-"*70)

    for model_name, accuracy in sorted(results_all.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name:<40s} {accuracy:>14.4f}")

    print("="*70)


def main():
    """主函数"""

    # 配置
    DATASET_PATH = "model/datasets/ST_VL53L8CX_handposture_dataset"
    TFLITE_DIR = "model/tflite"

    # 要评估的模型列表
    models_to_evaluate = [
        os.path.join(TFLITE_DIR, "hand_posture_float32.tflite"),
        os.path.join(TFLITE_DIR, "hand_posture_float16.tflite"),
        os.path.join(TFLITE_DIR, "hand_posture_int8.tflite"),
        os.path.join(TFLITE_DIR, "hand_posture_int8_full.tflite"),
    ]

    # 检查数据集是否存在
    if not os.path.exists(DATASET_PATH):
        print(f"错误: 数据集目录不存在: {DATASET_PATH}")
        return

    # 选择模式
    print("请选择评估模式:")
    print("1. 评估单个模型（详细分析）")
    print("2. 对比所有模型")

    mode = input("输入选项 (1 或 2): ").strip()

    if mode == "1":
        # 单个模型详细评估
        print("\n可用的模型:")
        for i, model_path in enumerate(models_to_evaluate, 1):
            exists = "✓" if os.path.exists(model_path) else "✗"
            print(f"  {i}. {os.path.basename(model_path)} {exists}")

        choice = input("\n选择要评估的模型编号: ").strip()
        try:
            model_idx = int(choice) - 1
            model_path = models_to_evaluate[model_idx]
        except (ValueError, IndexError):
            print("无效的选择，使用默认模型")
            model_path = models_to_evaluate[-1]  # 默认使用INT8 full

        if not os.path.exists(model_path):
            print(f"错误: 模型文件不存在: {model_path}")
            print("请先运行转换脚本生成TFLite模型")
            return

        # 创建验证器
        validator = TFLiteModelValidator(model_path, DATASET_PATH)

        # 加载数据集
        samples, labels = validator.load_dataset(max_samples_per_class=None)  # 加载全部数据

        # 评估
        results = validator.evaluate(samples, labels)

        # 打印结果
        validator.print_results(results)

        # 分析错误
        validator.analyze_errors(results, num_examples=10)

        # 保存结果
        output_path = model_path.replace('.tflite', '_evaluation.json')
        validator.save_results(results, output_path)

    elif mode == "2":
        # 对比多个模型
        existing_models = [m for m in models_to_evaluate if os.path.exists(m)]

        if len(existing_models) == 0:
            print("错误: 没有找到任何TFLite模型")
            print("请先运行转换脚本生成TFLite模型")
            return

        compare_models(existing_models, DATASET_PATH, max_samples_per_class=100)

    else:
        print("无效的选项")


if __name__ == "__main__":
    main()
