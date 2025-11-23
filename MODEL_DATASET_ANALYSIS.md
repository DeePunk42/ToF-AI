# ToF手势识别系统 - 模型与数据集完整分析

**生成时间**: 2025-11-23  
**项目**: Arduino ToF-AI 手势识别  
**传感器**: VL53L8CX (8x8 ToF)  
**模型**: CNN2D_ST_HandPosture_8classes

---

## 目录

1. [概述](#概述)
2. [模型架构分析](#模型架构分析)
3. [数据集分析](#数据集分析)
4. [数据格式详解](#数据格式详解)
5. [Arduino移植指南](#arduino移植指南)
6. [代码实现示例](#代码实现示例)

---

## 概述

本项目实现了基于VL53L8CX飞行时间(ToF)传感器的手势识别系统,使用轻量级CNN模型进行8种手势的实时分类。

### 关键特性

- **8种手势类别**: None, FlatHand, Like, Dislike, Fist, Love, BreakTime, CrossHands
- **传感器配置**: 8x8 zone (64个测量点)
- **模型大小**: ~31 KB (原始), ~7 KB (量化后)
- **推理速度**: ~1.5 ms @ 84MHz (STM32F401)
- **模型准确率**: 99.43% (验证集)

---

## 模型架构分析

### 模型文件信息

- **文件路径**: `model/CNN2D_ST_HandPosture_8classes.h5`
- **文件大小**: 31.09 KB
- **框架**: TensorFlow/Keras 2.8.0
- **总参数**: 2,752

### 网络层结构

| 层编号 | 层类型 | 输出形状 | 参数量 | 配置 |
|:------:|:------:|:--------:|:------:|:-----|
| 1 | InputLayer | (None, 8, 8, 2) | 0 | 输入层 |
| 2 | Conv2D | (None, 6, 6, 8) | 152 | 滤波器: 8, 卷积核: 3x3, 激活: linear |
| 3 | Activation | (None, 6, 6, 8) | 0 | ReLU |
| 4 | MaxPooling2D | (None, 3, 3, 8) | 0 | 池化: 2x2, 步长: 2 |
| 5 | Dropout | (None, 3, 3, 8) | 0 | 比率: 0.2 |
| 6 | Flatten | (None, 72) | 0 | 展平层 |
| 7 | Dense | (None, 32) | 2,336 | 神经元: 32, 激活: ReLU |
| 8 | Dense (输出) | (None, 8) | 264 | 神经元: 8, 激活: Softmax |

### 模型结构图

```
输入: (8×8×2)
    ↓
[Conv2D] 8个滤波器, 3×3
    ↓
[ReLU]
    ↓
[MaxPool] 2×2
    ↓
[Dropout] 20%
    ↓
[Flatten] → 72个特征
    ↓
[Dense] 32个神经元
    ↓
[Dense] 8个输出 (Softmax)
    ↓
输出: 8类概率
```

### 资源占用

| 平台 | Flash (KB) | RAM (KB) | 推理时间 |
|:----:|:----------:|:--------:|:--------:|
| STM32F401 @ 84MHz | 25.12 | 3.15 | 1.54 ms |
| STM32F401 (量化) | ~7 | ~2 | ~1.2 ms |

---

## 数据集分析

### 数据集概览

- **数据集路径**: `model/datasets/ST_VL53L8CX_handposture_dataset/`
- **总样本数**: 162
- **类别数**: 8
- **数据源**: 4个用户的多次采集

### 类别分布

| 类别 | 样本数 | 百分比 | 描述 |
|:----:|:------:|:------:|:-----|
| None | 10 | 6.17% | 无手势/背景 |
| FlatHand | 26 | 16.05% | 平手/手掌 |
| Like | 24 | 14.81% | 点赞(大拇指向上) |
| Dislike | 24 | 14.81% | 点踩(大拇指向下) |
| Fist | 35 | 21.60% | 拳头 |
| Love | 15 | 9.26% | 爱心手势 |
| BreakTime | 14 | 8.64% | 休息手势 |
| CrossHands | 14 | 8.64% | 双手交叉 |
| **总计** | **162** | **100%** | - |

### 数据采集信息

每个样本包含:
- **时间序列数据**: 可变长度(8-118帧)
- **空间分辨率**: 8×8 = 64个zone
- **测量频率**: ~10Hz (约100ms/帧)
- **用户数**: 4个不同用户
- **重复次数**: 每个用户每种手势2-6次

---

## 数据格式详解

### NPZ文件结构

每个样本以`.npz`格式存储,包含以下字段:

```python
{
    'start_tstmp': float64,           # 开始时间戳
    'end_tstmp': float64,             # 结束时间戳
    'zone_data': float64(4, 64, N),   # Zone数据 (4个通道, 64个zone, N个时间帧)
    'glob_data': float64(1, N),       # 全局数据
    'zone_head': str(4,),             # Zone字段名称
    'glob_head': str(1,)              # 全局字段名称
}
```

### Zone Data 通道

`zone_head` 包含4个字段:

1. **target_status**: 目标状态
2. **valid**: 数据有效性
3. **signal_per_spad**: 每个SPAD的信号强度 ⭐ (模型输入通道2)
4. **distance_mm**: 距离测量值(毫米) ⭐ (模型输入通道1)

### 数据统计

#### Distance (距离) 统计

| 类别 | 最小值 (mm) | 最大值 (mm) | 平均值 (mm) |
|:----:|:-----------:|:-----------:|:-----------:|
| None | 213 | 3781 | 633 |
| FlatHand | 147 | 4147 | 718 |
| Like | -481 | 3891 | 972 |
| Dislike | -522 | 3853 | 1001 |
| Fist | 190 | 3981 | 1083 |
| Love | 186 | 3974 | 849 |
| BreakTime | 304 | 3986 | 1152 |
| CrossHands | 266 | 3847 | 1199 |

#### Signal per SPAD 统计

| 类别 | 最小值 | 最大值 | 平均值 |
|:----:|:------:|:------:|:------:|
| None | 7 | 2682 | 441 |
| FlatHand | 6 | 4776 | 732 |
| Like | 5 | 3020 | 364 |
| Dislike | 6 | 2089 | 235 |
| Fist | 5 | 2141 | 247 |
| Love | 8 | 1914 | 442 |
| BreakTime | 5 | 959 | 149 |
| CrossHands | 6 | 968 | 136 |

### 数据示例: FlatHand 手势

**单帧8×8 Distance Map (mm)**:
```
[[2240 2218 2296 2290 2311 2340 2342  346]
 [2588 2599 2648 2731 2655 2628 2730 2640]
 [ 398  365 3055  394  331 3173 3171  362]
 [ 434  540  293  366  291  302  387  371]
 [ 391  560  332  324  301  308  330  366]
 [ 378  457  464  308  302  298  321  342]
 [ 369  278  354  302  292  290  333  336]
 [ 633  328  290  296  286  282  384  335]]
```

**单帧8×8 Signal Map**:
```
[[ 32  27  41  38  33  29  26  24]
 [ 20  24  21  18  25  21  21  19]
 [ 10  15  16  26  35  17  18  13]
 [  8  41 281 147 477  62  14  10]
 [ 10  70 314 321 589 114  21   9]
 [ 13 142 269 665 819 582  24  10]
 [ 25 670 262 1003 1087 529  21  18]
 [181 318 989 1132  972 375  19  15]]
```

> **观察**: 手掌区域(中心)显示较短距离(~300mm)和高信号强度(500-1000+),而背景区域显示较远距离(>2000mm)和低信号强度(<50)。

---

## Arduino移植指南

### 1. 硬件要求

#### 推荐平台

- **STM32F4系列** (推荐: STM32F401RE)
  - Flash: ≥128 KB
  - RAM: ≥32 KB
  - 时钟: 84 MHz
  
- **STM32L4系列** (低功耗)
  - Flash: ≥256 KB
  - RAM: ≥64 KB

- **Arduino平台** (实验性)
  - Arduino Due (Cortex-M3)
  - Arduino Portenta H7
  - ESP32-S3 (带充足RAM)

#### 传感器

- **VL53L8CX** ToF传感器
  - 8×8 zone配置
  - I2C接口 (400 kHz推荐)
  - 测量频率: 10-60 Hz

### 2. 软件依赖

```
TensorFlow Lite Micro (v2.8+)
VL53L8CX驱动库
STM32Cube HAL (STM32平台)
```

### 3. 数据输入规格

#### 输入张量

- **形状**: `(1, 8, 8, 2)`
- **数据类型**: `float32`
- **通道顺序**:
  - 通道0: `distance_mm` (距离,毫米)
  - 通道1: `signal_per_spad` (信号强度)

#### 数据采集

从VL53L8CX读取64个zone的数据:

```cpp
// 伪代码
VL53L8CX_ResultsData results;
vl53l8cx_get_ranging_data(&sensor, &results);

// 提取数据到模型输入缓冲区
for (int i = 0; i < 64; i++) {
    input_tensor[i] = results.distance_mm[i];        // 通道0
    input_tensor[64 + i] = results.signal_per_spad[i]; // 通道1
}
```

### 4. 数据预处理

#### 归一化公式

根据配置文件(`CNN2D_ST_HandPosture_8classes_config.yaml`):

```python
# 距离归一化
distance_normalized = (distance_mm - Min_distance) / (Max_distance - Min_distance)
distance_normalized = clip(distance_normalized, 0, 1)

# 参数
Max_distance = 400  # mm
Min_distance = 100  # mm
Background_distance = 120  # mm (背景阈值)
```

#### C/C++实现

```cpp
void preprocess_distance(float* distance, int size) {
    const float min_dist = 100.0f;
    const float max_dist = 400.0f;
    const float range = max_dist - min_dist;
    
    for (int i = 0; i < size; i++) {
        float normalized = (distance[i] - min_dist) / range;
        distance[i] = fmax(0.0f, fmin(1.0f, normalized)); // clip [0, 1]
    }
}

void preprocess_signal(float* signal, int size) {
    // 根据数据统计,signal范围约0-5000
    // 简单归一化到[0, 1]
    const float max_signal = 5000.0f;
    
    for (int i = 0; i < size; i++) {
        signal[i] = fmin(signal[i] / max_signal, 1.0f);
    }
}
```

### 5. 模型转换步骤

#### 步骤1: Keras → TensorFlow Lite

```python
import tensorflow as tf

# 加载Keras模型
model = tf.keras.models.load_model('CNN2D_ST_HandPosture_8classes.h5')

# 转换为TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# 保存
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

#### 步骤2: 量化(可选,减小模型大小)

```python
# INT8量化
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen  # 需要提供代表性数据集
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quantized_model = converter.convert()
```

#### 步骤3: TFLite → C数组

```bash
# 使用xxd工具
xxd -i model.tflite > model_data.cc
```

或使用Python脚本:

```python
def convert_to_c_array(tflite_model_path, output_path):
    with open(tflite_model_path, 'rb') as f:
        model_data = f.read()
    
    with open(output_path, 'w') as f:
        f.write('alignas(8) const unsigned char g_model[] = {\n')
        f.write(', '.join([f'0x{b:02x}' for b in model_data]))
        f.write('\n};\n')
        f.write(f'const int g_model_len = {len(model_data)};\n')

convert_to_c_array('model.tflite', 'model_data.cpp')
```

### 6. 输出解析

#### 输出张量

- **形状**: `(1, 8)`
- **数据类型**: `float32`
- **激活**: Softmax (值范围: 0-1, 总和≈1)

#### 类别索引映射

```cpp
const char* class_names[8] = {
    "None",        // 0
    "FlatHand",    // 1
    "Like",        // 2
    "Dislike",     // 3
    "Fist",        // 4
    "Love",        // 5
    "BreakTime",   // 6
    "CrossHands"   // 7
};
```

#### 推理结果处理

```cpp
// 获取输出张量
TfLiteTensor* output = interpreter->output(0);

// 找到最高置信度的类别
int max_index = 0;
float max_score = output->data.f[0];

for (int i = 1; i < 8; i++) {
    if (output->data.f[i] > max_score) {
        max_score = output->data.f[i];
        max_index = i;
    }
}

// 仅当置信度足够高时才接受结果
const float CONFIDENCE_THRESHOLD = 0.7f;
if (max_score > CONFIDENCE_THRESHOLD) {
    Serial.print("Detected: ");
    Serial.println(class_names[max_index]);
    Serial.print("Confidence: ");
    Serial.println(max_score);
}
```

---

## 代码实现示例

### 完整推理流程(Arduino/C++)

```cpp
#include <TensorFlowLite.h>
#include "model_data.h"  // 包含g_model数组
#include "vl53l8cx_api.h"

// TFLite全局对象
namespace {
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;
    
    // Tensor Arena (根据实际需要调整)
    constexpr int kTensorArenaSize = 8 * 1024;  // 8KB
    uint8_t tensor_arena[kTensorArenaSize];
}

// 类别名称
const char* CLASS_NAMES[8] = {
    "None", "FlatHand", "Like", "Dislike",
    "Fist", "Love", "BreakTime", "CrossHands"
};

// 初始化TFLite模型
bool setupModel() {
    // 加载模型
    model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model schema mismatch!");
        return false;
    }
    
    // 创建解释器
    static tflite::MicroMutableOpResolver<6> micro_op_resolver;
    micro_op_resolver.AddBuiltin(
        tflite::BuiltinOperator_CONV_2D,
        tflite::ops::micro::Register_CONV_2D());
    micro_op_resolver.AddBuiltin(
        tflite::BuiltinOperator_MAX_POOL_2D,
        tflite::ops::micro::Register_MAX_POOL_2D());
    micro_op_resolver.AddBuiltin(
        tflite::BuiltinOperator_RELU,
        tflite::ops::micro::Register_RELU());
    micro_op_resolver.AddBuiltin(
        tflite::BuiltinOperator_FULLY_CONNECTED,
        tflite::ops::micro::Register_FULLY_CONNECTED());
    micro_op_resolver.AddBuiltin(
        tflite::BuiltinOperator_SOFTMAX,
        tflite::ops::micro::Register_SOFTMAX());
    micro_op_resolver.AddBuiltin(
        tflite::BuiltinOperator_RESHAPE,
        tflite::ops::micro::Register_RESHAPE());
    
    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;
    
    // 分配张量
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.println("AllocateTensors() failed");
        return false;
    }
    
    // 获取输入输出指针
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    Serial.println("Model loaded successfully!");
    return true;
}

// 预处理函数
void preprocessData(float* distance, float* signal, float* input_buffer) {
    const float MIN_DIST = 100.0f;
    const float MAX_DIST = 400.0f;
    const float DIST_RANGE = MAX_DIST - MIN_DIST;
    const float MAX_SIGNAL = 5000.0f;
    
    // 填充输入张量: (8, 8, 2)
    for (int i = 0; i < 64; i++) {
        // 通道0: 归一化距离
        float norm_dist = (distance[i] - MIN_DIST) / DIST_RANGE;
        input_buffer[i * 2] = fmax(0.0f, fmin(1.0f, norm_dist));
        
        // 通道1: 归一化信号
        input_buffer[i * 2 + 1] = fmin(signal[i] / MAX_SIGNAL, 1.0f);
    }
}

// 执行推理
int runInference(VL53L8CX_ResultsData& results) {
    // 提取原始数据
    float distance[64];
    float signal[64];
    
    for (int i = 0; i < 64; i++) {
        distance[i] = results.distance_mm[i];
        signal[i] = results.signal_per_spad[i];
    }
    
    // 预处理并填充输入张量
    preprocessData(distance, signal, input->data.f);
    
    // 运行推理
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        Serial.println("Invoke failed!");
        return -1;
    }
    
    // 解析输出
    int predicted_class = 0;
    float max_score = output->data.f[0];
    
    for (int i = 1; i < 8; i++) {
        if (output->data.f[i] > max_score) {
            max_score = output->data.f[i];
            predicted_class = i;
        }
    }
    
    // 打印结果
    if (max_score > 0.7f) {  // 置信度阈值
        Serial.print("Gesture: ");
        Serial.print(CLASS_NAMES[predicted_class]);
        Serial.print(" (");
        Serial.print(max_score * 100);
        Serial.println("%)");
        
        // 打印所有类别的概率(调试用)
        Serial.println("All probabilities:");
        for (int i = 0; i < 8; i++) {
            Serial.print("  ");
            Serial.print(CLASS_NAMES[i]);
            Serial.print(": ");
            Serial.println(output->data.f[i] * 100);
        }
    }
    
    return predicted_class;
}

// Arduino主循环示例
void setup() {
    Serial.begin(115200);
    
    // 初始化ToF传感器
    // ... (VL53L8CX初始化代码)
    
    // 初始化模型
    if (!setupModel()) {
        while(1) {
            Serial.println("Model setup failed!");
            delay(1000);
        }
    }
}

void loop() {
    // 读取ToF数据
    VL53L8CX_ResultsData results;
    uint8_t status = vl53l8cx_get_ranging_data(&sensor, &results);
    
    if (status == VL53L8CX_STATUS_OK) {
        // 运行推理
        int gesture_class = runInference(results);
        
        // 处理识别结果
        // ...
    }
    
    delay(100);  // 10Hz采样率
}
```

### Python数据预处理示例

用于训练数据生成或验证:

```python
import numpy as np

def load_and_preprocess_sample(npz_file):
    """加载并预处理单个样本"""
    data = np.load(npz_file, allow_pickle=True)
    
    zone_data = data['zone_data']  # (4, 64, N)
    zone_head = data['zone_head']
    
    # 找到distance_mm和signal_per_spad的索引
    signal_idx = np.where(zone_head == 'signal_per_spad')[0][0]
    distance_idx = np.where(zone_head == 'distance_mm')[0][0]
    
    # 提取数据 (64, N)
    signal = zone_data[signal_idx, :, :]
    distance = zone_data[distance_idx, :, :]
    
    # 选择一帧(例如中间帧)
    frame_idx = signal.shape[1] // 2
    signal_frame = signal[:, frame_idx]  # (64,)
    distance_frame = distance[:, frame_idx]  # (64,)
    
    # 预处理
    distance_norm = np.clip((distance_frame - 100) / 300, 0, 1)
    signal_norm = np.clip(signal_frame / 5000, 0, 1)
    
    # 重塑为(8, 8, 2)
    input_tensor = np.stack([
        distance_norm.reshape(8, 8),
        signal_norm.reshape(8, 8)
    ], axis=-1)
    
    return input_tensor

# 批量处理
def prepare_dataset(dataset_path, class_names):
    X = []
    y = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_path, class_name)
        # 遍历所有npz文件
        for npz_file in glob.glob(f"{class_dir}/**/npz/*.npz", recursive=True):
            try:
                sample = load_and_preprocess_sample(npz_file)
                X.append(sample)
                y.append(class_idx)
            except Exception as e:
                print(f"Error processing {npz_file}: {e}")
    
    return np.array(X), np.array(y)
```

---

## 性能优化建议

### 1. 模型优化

- **量化**: INT8量化可减少模型大小约75%,轻微降低精度
- **剪枝**: 移除不重要的连接以减小模型
- **知识蒸馏**: 使用更大模型训练更小的学生模型

### 2. 推理优化

```cpp
// 使用固定点数学加速(INT8量化后)
#define USE_INT8_QUANTIZATION

// 减少推理频率(不是每帧都推理)
const int INFERENCE_INTERVAL_MS = 100;  // 10Hz

// 使用滑动窗口平滑结果
const int SMOOTHING_WINDOW = 3;
int gesture_history[SMOOTHING_WINDOW];
```

### 3. 传感器配置

```cpp
// VL53L8CX配置优化
vl53l8cx_set_ranging_frequency_hz(&sensor, 15);  // 15Hz测量
vl53l8cx_set_integration_time_ms(&sensor, 20);   // 20ms积分时间
vl53l8cx_set_target_order(&sensor, VL53L8CX_TARGET_ORDER_CLOSEST);
```

### 4. 功耗优化

- 使用低功耗模式在无手势时降低采样率
- 实现唤醒触发机制(例如距离变化阈值)
- 使用DMA传输减少CPU占用

---

## 故障排查

### 常见问题

#### 1. 推理结果不准确

**可能原因**:
- 数据未正确归一化
- 输入数据格式错误(通道顺序)
- 传感器校准问题

**解决方案**:
```cpp
// 验证输入数据范围
for (int i = 0; i < 128; i++) {
    if (input->data.f[i] < 0 || input->data.f[i] > 1.5) {
        Serial.print("Warning: input[");
        Serial.print(i);
        Serial.print("] = ");
        Serial.println(input->data.f[i]);
    }
}
```

#### 2. 内存不足

**症状**: `AllocateTensors()` 失败

**解决方案**:
- 增加 `kTensorArenaSize`
- 使用量化模型减少内存占用
- 检查是否有内存泄漏

#### 3. 推理速度慢

**可能原因**:
- CPU频率过低
- 未启用编译优化
- Tensor Arena碎片化

**解决方案**:
```cpp
// 编译选项优化
// platformio.ini
build_flags = 
    -O3
    -DNDEBUG
    -march=native
```

---

## 扩展资源

### 官方文档

- [STMicroelectronics Model Zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo)
- [VL53L8CX Datasheet](https://www.st.com/resource/en/datasheet/vl53l8cx.pdf)
- [TensorFlow Lite Micro Guide](https://www.tensorflow.org/lite/microcontrollers)
- [STM32Cube.AI](https://www.st.com/en/embedded-software/x-cube-ai.html)

### 相关论文

- CNN for ToF Gesture Recognition
- Efficient Deep Learning on Edge Devices
- Time-of-Flight Sensor Applications

### 工具

- **STM32CubeMX**: 配置STM32项目
- **STM32CubeIDE**: 集成开发环境
- **Netron**: 可视化神经网络模型
- **TensorBoard**: 模型训练可视化

---

## 许可证与引用

### 数据集许可

数据集来源于STMicroelectronics,使用需遵循相应许可协议。

### 引用

如果使用本项目,请引用:

```
ST Microelectronics. (2023). ST_VL53L8CX Hand Posture Dataset.
Model: CNN2D_ST_HandPosture_8classes.
Retrieved from: https://github.com/STMicroelectronics/stm32ai-modelzoo
```

---

## 附录

### A. 配置文件完整内容

完整的训练配置文件位于: `model/CNN2D_ST_HandPosture_8classes_config.yaml`

关键参数:
- `Max_distance`: 400 mm
- `Min_distance`: 100 mm
- `Background_distance`: 120 mm
- `batch_size`: 32
- `epochs`: 1000
- `learning_rate`: 0.01
- `dropout`: 0.2

### B. 硬件连接

#### STM32F4 + VL53L8CX

| VL53L8CX | STM32F4 |
|:--------:|:-------:|
| VCC | 3.3V |
| GND | GND |
| SDA | PB9 (I2C1_SDA) |
| SCL | PB8 (I2C1_SCL) |
| LPn | PA0 (GPIO) |
| I2C_RST | PA1 (GPIO) |

#### Arduino Due + VL53L8CX

| VL53L8CX | Arduino Due |
|:--------:|:-----------:|
| VCC | 3.3V |
| GND | GND |
| SDA | SDA (20) |
| SCL | SCL (21) |
| LPn | D7 |
| I2C_RST | D6 |

---

**文档版本**: 1.0  
**最后更新**: 2025-11-23  
**作者**: ToF-AI Project Team
