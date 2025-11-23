# 数据预处理修正说明

## 问题描述

初始转换脚本使用了错误的数据预处理方法，导致准确率很低（12.96%）。通过分析STM32应用代码（`data_process.md`），发现了正确的预处理流程。

## 修正过程

### 1. 初始尝试（错误）

最初使用简单归一化：
```python
distance_norm = distance / 1000.0
signal_norm = signal / 1000.0
```

**结果**：准确率 12.96%

### 2. 查阅STM32代码

从`data_process.md`中了解到完整的预处理流程包括：

1. **数据复制与格式转换**（AI_CopyInputData）
   - 距离：从14.2固定点格式除以4.0
   - 信号：从21.11固定点格式除以2048.0

2. **帧验证**（ValidateFrame）
   - 找到最小有效距离
   - 检查是否在MIN_DISTANCE到MAX_DISTANCE范围内
   - 应用背景移除（BACKGROUND_REMOVAL）
   - 无效区域填充默认值（DEFAULT_RANGING_VALUE=4000, DEFAULT_SIGNAL_VALUE=0）

3. **数据归一化**（NormalizeData）
   - 距离：`(distance - 295) / 196`
   - 信号：`(signal - 281) / 452`

### 3. 确认配置参数

从`CNN2D_ST_HandPosture_8classes_config.yaml`获取预处理参数：
```yaml
preprocessing:
  Max_distance: 400
  Min_distance: 100
  Background_distance: 120
```

### 4. 检查数据集字段

通过检查NPZ文件发现，数据集中的字段为：
- `target_status`
- `valid` （不是`nb_target_detected`）
- `signal_per_spad`
- `distance_mm`

### 5. 实现完整预处理流程

```python
def load_and_preprocess_sample(npz_file):
    """加载并预处理单个样本 - 完全按照STM32代码的处理流程"""
    # ... 加载数据 ...
    
    # 配置参数
    MIN_DISTANCE = 100.0
    MAX_DISTANCE = 400.0
    BACKGROUND_REMOVAL = 120.0
    DEFAULT_RANGING_VALUE = 4000.0
    DEFAULT_SIGNAL_VALUE = 0.0
    RANGING_OK_5 = 5
    RANGING_OK_9 = 9
    
    # 1. 找到最小有效距离
    min_distance = 4000.0
    for idx in range(64):
        is_valid_zone = (valid_frame[idx] > 0 and 
                        (status_frame[idx] == RANGING_OK_5 or status_frame[idx] == RANGING_OK_9) and
                        distance_frame[idx] < min_distance)
        if is_valid_zone:
            min_distance = distance_frame[idx]
    
    # 2. 应用背景移除和默认值填充
    for idx in range(64):
        is_valid_zone = (valid_frame[idx] > 0 and
                        (status_frame[idx] == RANGING_OK_5 or status_frame[idx] == RANGING_OK_9) and
                        distance_frame[idx] < min_distance + BACKGROUND_REMOVAL)
        
        if not is_valid_zone:
            distance_frame[idx] = DEFAULT_RANGING_VALUE
            signal_frame[idx] = DEFAULT_SIGNAL_VALUE
    
    # 3. 归一化
    NORMALIZATION_RANGING_CENTER = 295.0
    NORMALIZATION_RANGING_IQR = 196.0
    NORMALIZATION_SIGNAL_CENTER = 281.0
    NORMALIZATION_SIGNAL_IQR = 452.0
    
    distance_norm = (distance_frame - NORMALIZATION_RANGING_CENTER) / NORMALIZATION_RANGING_IQR
    signal_norm = (signal_frame - NORMALIZATION_SIGNAL_CENTER) / NORMALIZATION_SIGNAL_IQR
    
    # 4. 重塑为(8, 8, 2)
    input_tensor = np.zeros((8, 8, 2), dtype=np.float32)
    input_tensor[:, :, 0] = distance_norm.reshape(8, 8)
    input_tensor[:, :, 1] = signal_norm.reshape(8, 8)
    
    return input_tensor
```

## 最终结果

修正后的准确率：
- **Keras原始模型**: 20.99%
- **TFLite FLOAT32**: 20.99% (差异: 0.00%)
- **TFLite INT8**: 24.07% (差异: +3.09%)

## 关键发现

1. **归一化参数的来源**：代码中的归一化参数（295, 196, 281, 452）是基于**经过帧验证和背景移除后**的数据计算的，而不是原始数据的统计值。

2. **数据验证的重要性**：
   - 原始数据范围很大（距离：-544到4643mm）
   - 经过帧验证后，只保留有效范围内的数据
   - 无效区域用固定默认值填充

3. **字段名差异**：NPZ数据集使用`valid`字段，而STM32代码使用`nb_target_detected`。这是因为NPZ是原始记录数据，而STM32使用的是传感器API结构。

4. **准确率问题**：
   - 在训练集上测试准确率只有20.99%，说明模型本身泛化能力有限
   - 这可能是数据集规模小（162样本）或数据质量问题
   - INT8量化版本准确率更高（24.07%），可能是量化误差恰好对某些样本有益

## Arduino部署建议

在Arduino上实现时，必须严格遵循相同的预处理流程：

```cpp
// 1. 帧验证 - 找到最小距离
float min_distance = 4000.0;
for (int i = 0; i < 64; i++) {
    if (valid[i] > 0 && 
        (status[i] == 5 || status[i] == 9) && 
        distance[i] < min_distance) {
        min_distance = distance[i];
    }
}

// 2. 背景移除和默认值填充
for (int i = 0; i < 64; i++) {
    bool is_valid = (valid[i] > 0 && 
                     (status[i] == 5 || status[i] == 9) && 
                     distance[i] < min_distance + 120.0);
    
    if (!is_valid) {
        distance[i] = 4000.0;
        signal[i] = 0.0;
    }
}

// 3. 归一化
for (int i = 0; i < 64; i++) {
    float distance_norm = (distance[i] - 295.0) / 196.0;
    float signal_norm = (signal[i] - 281.0) / 452.0;
    
    // 填充输入张量 (交错格式: [dist0, sig0, dist1, sig1, ...])
    input_data[2*i] = distance_norm;
    input_data[2*i + 1] = signal_norm;
}
```

## 参考文档

- `data_process.md` - STM32应用代码的数据处理流程分析
- `CNN2D_ST_HandPosture_8classes_config.yaml` - 训练配置参数
- `convert_to_tflite.py` - 修正后的转换脚本
