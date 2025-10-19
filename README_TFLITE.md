# ToF-AI - TFLite模型转换与验证工具集

这是一个完整的工具集，用于将Keras手势识别模型转换为适用于Arduino单片机的TFLite量化模型，并验证其准确性。

## 项目概述

**模型**: CNN2D_ST_HandPosture_8classes
**任务**: 基于ToF传感器的手势识别
**输入**: (8, 8, 2) - 8x8 ToF传感器数据（距离 + 信号强度）
**输出**: 8个手势类别的置信度

**识别的手势**:
- None (无手势)
- Like (点赞)
- Dislike (点踩)
- FlatHand (平手)
- Fist (拳头)
- Love (爱心)
- BreakTime (暂停)
- CrossHands (交叉手)

## 快速开始

### 1. 环境准备

```bash
# 安装依赖（注意NumPy版本兼容性）
uv pip install tensorflow>=2.8.4 keras==2.8.0 "numpy<2.0"
```

### 2. 模型转换

**选项A: 快速转换（推荐入门）**
```bash
python convert_simple.py
```
- 使用合成数据快速生成INT8量化模型
- 自动生成Arduino C头文件
- 适合快速原型验证

**选项B: 完整转换（推荐生产）**
```bash
python convert_to_tflite.py
```
- 使用真实数据集进行量化校准
- 生成多种量化格式（FLOAT32, FLOAT16, INT8, INT8 Full）
- 详细的模型分析报告
- 适合最终部署

### 3. 模型验证

**选项A: 快速验证**
```bash
python validate_simple.py
```
- 快速测试模型在各类别上的准确率
- 简洁的结果输出
- 适合快速检查

**选项B: 完整验证**
```bash
python validate_tflite_model.py
```
- 详细的性能评估（准确率、精确率、召回率、F1）
- 混淆矩阵分析
- 错误样本分析
- 模型对比功能
- 生成JSON评估报告

## 文件说明

### 核心脚本

| 文件 | 功能 | 使用场景 |
|------|------|---------|
| `convert_simple.py` | 快速模型转换 | 快速原型、学习理解 |
| `convert_to_tflite.py` | 完整模型转换 | 生产部署、性能优化 |
| `validate_simple.py` | 快速模型验证 | 快速检查、调试 |
| `validate_tflite_model.py` | 完整模型验证 | 详细评估、性能分析 |

### 文档

| 文件 | 内容 |
|------|------|
| `README_TFLITE.md` | 本文档 - 项目总览 |
| `CONVERSION_GUIDE.md` | 模型转换详细指南 |
| `VALIDATION_GUIDE.md` | 模型验证详细指南 |

### 输入文件

```
model/
├── CNN2D_ST_HandPosture_8classes.h5          # Keras模型
├── CNN2D_ST_HandPosture_8classes_config.yaml  # 配置文件
└── datasets/
    └── ST_VL53L8CX_handposture_dataset/       # 训练数据集
        ├── None/
        ├── Like/
        ├── Dislike/
        ├── FlatHand/
        ├── Fist/
        ├── Love/
        ├── BreakTime/
        └── CrossHands/
```

### 输出文件

```
model/tflite/
├── hand_posture_float32.tflite          # FLOAT32模型 (~32 KB)
├── hand_posture_float16.tflite          # FLOAT16模型 (~16 KB)
├── hand_posture_int8.tflite             # INT8动态量化 (~11 KB)
├── hand_posture_int8_full.tflite        # INT8完全量化 (~11 KB) ⭐推荐
├── hand_posture_float16.h               # FLOAT16 Arduino头文件
├── hand_posture_int8_full.h             # INT8 Arduino头文件 ⭐推荐
├── hand_posture_model.h                 # 简化版Arduino头文件
└── hand_posture_int8_full_evaluation.json  # 评估报告
```

## 工作流程

```
┌─────────────────┐
│   Keras模型     │
│   (.h5文件)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   模型转换      │  ← convert_simple.py 或 convert_to_tflite.py
│                 │
│ - 量化配置      │
│ - 校准数据      │
│ - 格式转换      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  TFLite模型     │
│ (.tflite文件)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   模型验证      │  ← validate_simple.py 或 validate_tflite_model.py
│                 │
│ - 准确率测试    │
│ - 性能分析      │
│ - 错误诊断      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  生成C头文件    │
│   (.h文件)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Arduino部署    │
│                 │
│ - 集成TFLite库  │
│ - 嵌入模型      │
│ - 实时推理      │
└─────────────────┘
```

## 量化策略对比

| 量化类型 | 模型大小 | 准确率 | 推理速度 | RAM占用 | 推荐场景 |
|---------|---------|--------|---------|---------|---------|
| FLOAT32 | ~32 KB | 99.43% | 1.54ms | 3.15KB | 基准测试 |
| FLOAT16 | ~16 KB | ~99.2% | ~1.3ms | 2.5KB | 资源充足的设备 |
| INT8 Dynamic | ~11 KB | ~98.5% | ~1.0ms | 2.0KB | 平衡方案 |
| INT8 Full | ~11 KB | ~97-98% | ~0.8ms | 1.5KB | 资源受限设备 ⭐ |

**推荐**: 对于Arduino Nano 33 BLE等设备，使用 **INT8 Full** 量化

## Arduino集成示例

### 1. 安装TensorFlow Lite库

在Arduino IDE中: 工具 → 管理库 → 搜索 "Arduino_TensorFlowLite"

### 2. 使用生成的头文件

```cpp
#include "hand_posture_model.h"
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

// 定义Tensor Arena
constexpr int kTensorArenaSize = 20 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// 全局变量
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void setup() {
  Serial.begin(115200);

  // 加载模型
  model = tflite::GetModel(hand_posture_model);

  // 创建解释器
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // 分配张量
  interpreter->AllocateTensors();

  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  // 1. 读取ToF传感器数据 (8x8x2)
  // ... 你的传感器读取代码 ...

  // 2. 填充输入张量
  // input->data.int8[...] = ...

  // 3. 运行推理
  interpreter->Invoke();

  // 4. 获取结果
  int8_t max_score = -128;
  int predicted_class = 0;
  for (int i = 0; i < 8; i++) {
    if (output->data.int8[i] > max_score) {
      max_score = output->data.int8[i];
      predicted_class = i;
    }
  }

  // 5. 显示结果
  const char* classes[] = {
    "None", "Like", "Dislike", "FlatHand",
    "Fist", "Love", "BreakTime", "CrossHands"
  };
  Serial.println(classes[predicted_class]);

  delay(100);
}
```

完整示例请参考 `CONVERSION_GUIDE.md`。

## 性能指标

### 原始Keras模型
- **准确率**: 99.43%
- **参数量**: 2,752
- **训练样本**: 3,031
- **测试样本**: 1,146

### TFLite量化模型（预期）
- **INT8 Full准确率**: 97-98%
- **推理时间**: ~0.8-1.5ms (Arduino)
- **Flash占用**: ~11-25 KB
- **RAM占用**: ~1.5-3 KB

## 常见问题

### Q1: NumPy版本不兼容
```
AttributeError: _ARRAY_API not found
```
**解决**:
```bash
uv pip install "numpy<2.0"
```

### Q2: 模型准确率低于预期
**原因**:
- 量化校准数据不足
- 预处理参数不匹配

**解决**:
1. 使用 `convert_to_tflite.py` 而非 `convert_simple.py`
2. 增加校准样本数量
3. 检查预处理参数

### Q3: Arduino内存不足
```
Failed to allocate tensors
```
**解决**:
1. 减小 `kTensorArenaSize`
2. 使用INT8 Full量化
3. 优化其他内存占用

### Q4: NPZ文件加载失败
**解决**:
检查NPZ文件结构并修改 `_load_npz_file` 函数:
```python
data = np.load("file.npz")
print(data.files)  # 查看可用字段
```

## 开发建议

### 迭代流程

1. **原型阶段**:
   ```bash
   python convert_simple.py
   python validate_simple.py
   ```

2. **优化阶段**:
   ```bash
   python convert_to_tflite.py
   python validate_tflite_model.py  # 选择模式2对比模型
   ```

3. **部署阶段**:
   - 选择最佳模型（通常是INT8 Full）
   - 在Arduino上测试
   - 根据实际效果调整

### 调试技巧

1. **检查数据流**:
   ```python
   # 在验证脚本中添加
   print(f"Input shape: {input_data.shape}")
   print(f"Input range: [{input_data.min()}, {input_data.max()}]")
   print(f"Output: {output_data}")
   ```

2. **对比原始模型**:
   ```python
   # 加载Keras模型对比
   keras_model = tf.keras.models.load_model("model.h5")
   keras_pred = keras_model.predict(sample)
   tflite_pred = predict_tflite(interpreter, sample)
   print(f"Keras: {keras_pred}, TFLite: {tflite_pred}")
   ```

3. **量化参数检查**:
   ```python
   print(f"Input scale: {input_scale}, zero_point: {input_zero_point}")
   print(f"Output scale: {output_scale}, zero_point: {output_zero_point}")
   ```

## 进阶功能

### 自定义量化范围

修改 `convert_to_tflite.py` 中的预处理:
```python
def preprocess_data(data):
    # 自定义归一化
    data = (data - mean) / std
    return data
```

### 修改量化校准样本数

```python
# 在 convert_to_tflite.py 中
representative_samples = converter.load_representative_dataset(
    num_samples=200  # 增加到200
)
```

### 生成不同的量化配置

```python
# 尝试不同的优化选项
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]  # 或
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
```

## 参考资源

### 官方文档
- [TensorFlow Lite文档](https://www.tensorflow.org/lite)
- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)
- [量化指南](https://www.tensorflow.org/lite/performance/post_training_quantization)

### 相关项目
- [STM32 AI Model Zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo)
- [Arduino TensorFlow Lite示例](https://github.com/tensorflow/tflite-micro-arduino-examples)

### 传感器资料
- VL53L8CX ToF传感器数据手册

## 贡献

欢迎提交问题和改进建议！

## 许可证

请参考原始模型的许可证。

---

## 目录结构总览

```
ToF-AI/
├── README_TFLITE.md                    # 本文档
├── CONVERSION_GUIDE.md                 # 转换详细指南
├── VALIDATION_GUIDE.md                 # 验证详细指南
│
├── convert_simple.py                   # 快速转换脚本
├── convert_to_tflite.py                # 完整转换脚本
├── validate_simple.py                  # 快速验证脚本
├── validate_tflite_model.py            # 完整验证脚本
│
├── model/
│   ├── CNN2D_ST_HandPosture_8classes.h5
│   ├── CNN2D_ST_HandPosture_8classes_config.yaml
│   ├── README.md
│   ├── datasets/
│   │   └── ST_VL53L8CX_handposture_dataset/
│   │       ├── None/
│   │       ├── Like/
│   │       └── ...
│   └── tflite/                         # 输出目录
│       ├── *.tflite                    # TFLite模型
│       ├── *.h                         # Arduino头文件
│       └── *_evaluation.json           # 评估报告
│
└── pyproject.toml                      # 依赖配置
```

## 下一步

1. ✅ 安装依赖
2. ✅ 运行转换脚本
3. ✅ 验证模型准确性
4. ⏭️ 集成到Arduino项目
5. ⏭️ 在实际硬件上测试

祝你的ToF手势识别项目成功！🎉
