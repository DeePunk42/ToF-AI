# ToF-AI: Hand Gesture Recognition for Arduino/STM32

基于VL53L8CX ToF传感器的嵌入式手势识别系统,使用轻量级CNN模型实现8种手势的实时分类。

## 项目概述

本项目将ST的CNN2D_ST_HandPosture模型移植到Arduino/STM32平台,实现基于飞行时间(ToF)传感器的手势识别功能。

### 特性

- 🎯 **8种手势识别**: None, FlatHand, Like, Dislike, Fist, Love, BreakTime, CrossHands
- 🚀 **快速推理**: ~1.5ms @ 84MHz (STM32F401)
- 💾 **小型模型**: 仅31KB (原始) / 7KB (量化)
- 📊 **高准确率**: 99.43% (验证集)
- 🔧 **易于部署**: 完整的移植指南和代码示例

## 文档导航

### 📚 主要文档

1. **[SUMMARY_CN.md](./SUMMARY_CN.md)** - 📄 **快速总结** (推荐首先阅读)
   - 模型和数据集的关键信息
   - 快速参考表格
   - 下一步指南

2. **[MODEL_DATASET_ANALYSIS.md](./MODEL_DATASET_ANALYSIS.md)** - 📖 **完整技术文档**
   - 详细的模型架构分析
   - 数据集结构和统计
   - Arduino移植完整指南
   - 代码实现示例
   - 故障排查和优化建议

3. **[CONVERSION_REPORT.md](./CONVERSION_REPORT.md)** - 🔄 **模型转换报告** ⭐
   - Keras到TFLite转换结果
   - 准确率验证报告
   - 正确的数据预处理方法
   - Arduino部署指南

4. **[model/README.md](./model/README.md)** - 🔍 **原始模型说明**
   - ST官方模型文档
   - 性能指标
   - 训练信息

### 🛠️ 工具脚本

- **[analyze_dataset_v2.py](./analyze_dataset_v2.py)** - 分析模型和数据集结构
- **[convert_to_tflite.py](./convert_to_tflite.py)** - ⭐ Keras转TFLite并验证准确性
- **[test_preprocessing.py](./test_preprocessing.py)** - 测试数据预处理方法
- **[visualize_data.py](./visualize_data.py)** - 可视化ToF数据(可选)

### 📊 生成的报告

- **[analysis_report.txt](./analysis_report.txt)** - 数据集分析完整输出
- **[conversion_report.txt](./conversion_report.txt)** - 模型转换详细日志

### 📁 项目结构

```
ToF-AI/
├── README.md                          # 本文件
├── SUMMARY_CN.md                      # 快速总结 ⭐
├── MODEL_DATASET_ANALYSIS.md          # 完整技术文档 ⭐
├── pyproject.toml                     # Python项目配置
├── analyze_dataset_v2.py              # 数据分析脚本
├── analysis_report.txt                # 分析报告
│
├── model/                             # 模型文件夹
│   ├── README.md                      # 模型说明
│   ├── CNN2D_ST_HandPosture_8classes.h5         # Keras模型 ⭐
│   ├── CNN2D_ST_HandPosture_8classes_config.yaml # 训练配置
│   └── datasets/                      # 数据集
│       └── ST_VL53L8CX_handposture_dataset/
│           ├── None/                  # 8个类别文件夹
│           ├── FlatHand/
│           ├── Like/
│           ├── Dislike/
│           ├── Fist/
│           ├── Love/
│           ├── BreakTime/
│           └── CrossHands/
│
└── main/                              # Arduino主程序
    ├── main.ino                       # Arduino sketch
    ├── ml.cpp / ml.h                  # 机器学习推理
    ├── model.cpp / model.h            # 模型数据
    └── ToF.cpp / ToF.h                # ToF传感器驱动
```

## 快速开始

### 1. 查看分析报告

```bash
# 首先阅读快速总结
cat SUMMARY_CN.md

# 或查看完整文档
cat MODEL_DATASET_ANALYSIS.md
```

### 2. 运行数据分析(可选)

```bash
# 安装依赖
pip install tensorflow==2.8.4 keras==2.8.0 numpy

# 运行分析脚本
python analyze_dataset_v2.py
```

### 3. 模型转换

✅ **已完成!** 运行转换脚本生成TFLite模型和C数组:

```bash
python convert_to_tflite.py
```

**生成的文件**:
- `model_float32.tflite` (13.21 KB) - FLOAT32格式
- `model_int8.tflite` (5.91 KB) - INT8量化格式 ⭐ 推荐
- `model_data_float32.h` - FLOAT32 C数组
- `model_data_int8.h` - INT8 C数组 ⭐ 推荐

**转换结果**:
- 模型大小减少 81% (相比原始Keras)
- INT8量化无明显精度损失
- 包含完整的准确率验证

### 4. Arduino部署

详细的部署步骤请参考 [MODEL_DATASET_ANALYSIS.md](./MODEL_DATASET_ANALYSIS.md) 的 "Arduino移植指南" 部分。

## 模型信息速览

| 项目 | 值 |
|:-----|:---|
| **模型文件** | CNN2D_ST_HandPosture_8classes.h5 |
| **模型大小** | 31 KB (FLOAT32) / 7 KB (INT8) |
| **总参数** | 2,752 |
| **输入形状** | (8, 8, 2) |
| **输出形状** | (8,) - 8个类别概率 |
| **准确率** | 99.43% |
| **推理时间** | 1.5ms @ 84MHz |

## 数据集信息速览

| 项目 | 值 |
|:-----|:---|
| **数据集路径** | model/datasets/ST_VL53L8CX_handposture_dataset/ |
| **总样本数** | 162 |
| **类别数** | 8 |
| **传感器** | VL53L8CX (8×8 ToF) |
| **数据通道** | distance_mm, signal_per_spad |

### 类别分布

| 类别 | 样本数 | 百分比 |
|:----:|:------:|:------:|
| Fist | 35 | 21.60% |
| FlatHand | 26 | 16.05% |
| Like | 24 | 14.81% |
| Dislike | 24 | 14.81% |
| Love | 15 | 9.26% |
| BreakTime | 14 | 8.64% |
| CrossHands | 14 | 8.64% |
| None | 10 | 6.17% |

## 硬件要求

### 推荐平台

- **STM32F4系列** (推荐: STM32F401RE)
  - Flash: ≥128 KB
  - RAM: ≥32 KB
  - 时钟: 84 MHz

- **Arduino平台**
  - Arduino Due
  - Arduino Portenta H7
  - ESP32-S3

### 传感器

- **VL53L8CX** ToF传感器
  - 8×8 zone配置
  - I2C接口
  - 测量频率: 10-60 Hz

## 技术栈

- **框架**: TensorFlow/Keras 2.8.0
- **部署**: TensorFlow Lite Micro
- **语言**: Python 3.10 (分析), C/C++ (部署)
- **平台**: Arduino, STM32

## 相关资源

### 官方资源

- [STM32 AI Model Zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo)
- [VL53L8CX产品页面](https://www.st.com/en/imaging-and-photonics-solutions/vl53l8cx.html)
- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)

### 文档和教程

- [STM32Cube.AI文档](https://www.st.com/en/embedded-software/x-cube-ai.html)
- [VL53L8CX Arduino驱动](https://github.com/stm32duino/VL53L8CX)

## 许可证

本项目基于ST的预训练模型和数据集。请遵循相应的开源许可证。

- 模型: ST Microelectronics
- 数据集: ST_VL53L8CX_handposture_dataset

## 贡献

欢迎提交问题和改进建议!

## 联系方式

- **项目仓库**: [GitHub - ToF-AI](https://github.com/DeePunk42/ToF-AI)
- **问题反馈**: 请使用GitHub Issues

---

**最后更新**: 2025-11-23  
**版本**: 1.0.0
