# TFLite模型转换指南

本指南介绍如何将Keras模型量化转换为适用于Arduino单片机的TFLite模型。

## 项目信息

### 模型详情
- **模型名称**: CNN2D_ST_HandPosture_8classes
- **任务**: 手势识别 (Hand Posture Recognition)
- **输入**: (8, 8, 2) - ToF传感器数据
  - 第1通道: 距离 (distance_mm)
  - 第2通道: 信号强度 (signal_per_spad)
- **输出**: 8个类别的置信度
- **参数量**: 2,752

### 识别类别
| 索引 | 类别名称 | 说明 |
|------|---------|------|
| 0 | None | 无手势 |
| 1 | Like | 点赞 |
| 2 | Dislike | 点踩 |
| 3 | FlatHand | 平手 |
| 4 | Fist | 拳头 |
| 5 | Love | 爱心 |
| 6 | BreakTime | 暂停 |
| 7 | CrossHands | 交叉手 |

## 转换脚本说明

### 方案1: 完整转换脚本 (推荐)

**文件**: `convert_to_tflite.py`

**功能**:
- 从数据集加载真实的代表性样本用于量化校准
- 生成4种不同量化级别的模型
- 生成Arduino C头文件
- 详细的模型分析

**使用方法**:
```bash
python convert_to_tflite.py
```

**输出文件** (在 `model/tflite/` 目录):
1. `hand_posture_float32.tflite` - FLOAT32基准模型 (~32 KB)
2. `hand_posture_float16.tflite` - FLOAT16量化 (~16 KB, 推荐用于资源较多的Arduino)
3. `hand_posture_int8.tflite` - INT8动态范围量化 (~11 KB)
4. `hand_posture_int8_full.tflite` - 完全INT8量化 (~11 KB, 推荐用于资源受限的Arduino)
5. `hand_posture_float16.h` - FLOAT16 C头文件
6. `hand_posture_int8_full.h` - INT8 C头文件

### 方案2: 简化转换脚本 (快速)

**文件**: `convert_simple.py`

**功能**:
- 使用合成数据快速转换
- 直接生成INT8完全量化模型
- 生成Arduino C头文件

**使用方法**:
```bash
python convert_simple.py
```

**输出文件** (在 `model/tflite/` 目录):
1. `hand_posture_int8_full.tflite` - 完全INT8量化模型
2. `hand_posture_model.h` - Arduino C头文件

## 环境准备

### 修复NumPy兼容性问题

当前项目使用的TensorFlow版本与NumPy 2.x不兼容，需要降级NumPy：

```bash
# 方法1: 使用uv (推荐)
uv pip install "numpy<2.0"

# 方法2: 使用pip
pip install "numpy<2.0"
```

### 完整依赖安装

```bash
uv pip install tensorflow>=2.8.4 keras==2.8.0 "numpy<2.0"
```

## Arduino集成指南

### 1. 安装TensorFlow Lite库

在Arduino IDE中:
1. 打开 **工具** > **管理库**
2. 搜索 "Arduino_TensorFlowLite"
3. 安装最新版本

### 2. 使用生成的头文件

将生成的 `.h` 文件复制到Arduino项目目录，然后在代码中引用：

```cpp
#include "hand_posture_model.h"

// 包含TensorFlow Lite库
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

// 定义Tensor Arena大小 (根据需要调整)
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

  // 创建操作解析器
  static tflite::AllOpsResolver resolver;

  // 创建解释器
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // 分配张量
  interpreter->AllocateTensors();

  // 获取输入输出张量指针
  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Model loaded successfully!");
}

void loop() {
  // 1. 从ToF传感器读取8x8数据
  float distance[8][8];
  float signal[8][8];
  // ... 读取传感器数据的代码 ...

  // 2. 将数据填充到输入张量
  // 注意: INT8模型需要量化输入
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      // 归一化并量化到INT8范围 [-128, 127]
      int idx = (i * 8 + j) * 2;
      input->data.int8[idx] = quantize_float_to_int8(distance[i][j]);
      input->data.int8[idx + 1] = quantize_float_to_int8(signal[i][j]);
    }
  }

  // 3. 运行推理
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return;
  }

  // 4. 读取输出 (8个类别的置信度)
  int8_t max_score = -128;
  int max_index = 0;

  for (int i = 0; i < 8; i++) {
    int8_t score = output->data.int8[i];
    if (score > max_score) {
      max_score = score;
      max_index = i;
    }
  }

  // 5. 打印结果
  const char* classes[] = {
    "None", "Like", "Dislike", "FlatHand",
    "Fist", "Love", "BreakTime", "CrossHands"
  };

  Serial.print("Detected: ");
  Serial.println(classes[max_index]);

  delay(100);
}

// INT8量化辅助函数
int8_t quantize_float_to_int8(float value) {
  // 根据训练时的量化参数调整
  // 这里需要从模型的量化信息中获取scale和zero_point
  // 公式: quantized = value / scale + zero_point
  // 示例 (需要根据实际量化参数调整):
  return (int8_t)constrain((int)(value * 0.1f), -128, 127);
}
```

### 3. 内存要求

| Arduino板 | RAM | 推荐模型 | Tensor Arena大小 |
|----------|-----|---------|-----------------|
| Arduino Nano 33 BLE | 256 KB | INT8 Full | 20-30 KB |
| Arduino Portenta H7 | 1 MB | FLOAT16 | 30-40 KB |
| ESP32 | 520 KB | INT8 Full | 20-30 KB |

## 量化类型对比

| 类型 | 模型大小 | 精度 | 推理速度 | 适用场景 |
|------|---------|------|---------|---------|
| FLOAT32 | ~32 KB | 最高 | 较慢 | 基准测试 |
| FLOAT16 | ~16 KB | 高 | 中等 | 资源充足的设备 |
| INT8 | ~11 KB | 中等 | 快 | 通用微控制器 |
| INT8 Full | ~11 KB | 中等 | 最快 | 资源受限设备 (推荐) |

## 性能参考

根据ST官方数据 (STM32F401 @ 84MHz):
- **推理时间**: ~1.54 ms
- **准确率**: 99.43% (在测试集上)
- **Flash占用**: ~25 KB
- **RAM占用**: ~3.15 KB

## 数据预处理

根据配置文件 (`CNN2D_ST_HandPosture_8classes_config.yaml`):

```python
# 预处理参数
Max_distance = 400      # 最大距离 (mm)
Min_distance = 100      # 最小距离 (mm)
Background_distance = 120  # 背景距离 (mm)

# 在Arduino中应用相同的预处理
def preprocess_distance(raw_distance):
    if raw_distance > Max_distance:
        return Max_distance
    elif raw_distance < Min_distance:
        return Background_distance
    return raw_distance
```

## 故障排除

### 问题1: NumPy版本不兼容
```
AttributeError: _ARRAY_API not found
```
**解决方案**: 降级NumPy到1.x版本
```bash
uv pip install "numpy<2.0"
```

### 问题2: Arduino内存不足
```
Failed to allocate tensors
```
**解决方案**:
1. 减小 `kTensorArenaSize`
2. 使用INT8 Full量化模型
3. 关闭其他不必要的功能

### 问题3: 推理结果不准确
**解决方案**:
1. 检查输入数据的量化参数是否正确
2. 验证预处理步骤与训练时一致
3. 使用真实数据集进行量化校准 (使用 `convert_to_tflite.py`)

## 进一步优化

1. **使用真实数据集校准**: 运行 `convert_to_tflite.py` 而不是 `convert_simple.py`
2. **调整Tensor Arena大小**: 根据实际内存使用情况优化
3. **性能分析**: 使用Arduino串口监控推理时间
4. **功耗优化**: 在推理间隙使用低功耗模式

## 参考资源

- [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
- [Arduino TensorFlow Lite库](https://github.com/tensorflow/tflite-micro-arduino-examples)
- [STM32 AI Model Zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo)

## 许可证

请参考原始模型的许可证信息。
