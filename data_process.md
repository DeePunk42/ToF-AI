在hand_posture项目中，输入数据从传感器读取到提交给模型经历了以下处理流程：

## 整体数据流程

在主循环中，数据处理按以下顺序进行： [1](#0-0) 

## 1. 传感器数据读取

**`Sensor_GetRangingData()`** 函数从VL53LMZ ToF传感器获取原始测距数据： [2](#0-1) 

此函数等待传感器中断，然后调用 `vl53lmz_get_ranging_data()` 获取包含距离和信号强度的原始数据。

## 2. 数据预处理（Network_Preprocess）

预处理包含三个关键步骤：

### 2.1 数据复制与格式转换（AI_CopyInputData） [3](#0-2) 

这一步将传感器的固定点格式数据转换为浮点数：
- **距离数据**：从14.2固定点格式除以4.0转换为浮点数（mm）
- **信号强度**：从21.11固定点格式除以2048.0转换为浮点数
- 还可选择性地旋转数据（通过SENSOR_ROTATION_180宏）

### 2.2 帧验证（ValidateFrame） [4](#0-3) 

验证步骤包括：
- 找到所有有效区域中的**最小距离**
- 检查最小距离是否在有效范围内（MIN_DISTANCE 到 MAX_DISTANCE之间）
- 对于无效的区域，使用**默认值**填充（DEFAULT_RANGING_VALUE=4000, DEFAULT_SIGNAL_VALUE=0）
- 应用**背景移除**：只保留距离在 `min + BACKGROUND_REMOVAL` 范围内的目标

### 2.3 数据归一化（NormalizeData） [5](#0-4) 

归一化使用预定义的统计参数：
- **距离归一化**：`(ranging - 295) / 196`
- **信号归一化**：`(peak - 281) / 452`
- 输出交错存储：[距离0, 信号0, 距离1, 信号1, ...]

归一化参数定义在： [6](#0-5) 

## 3. 模型推理（Network_Inference） [7](#0-6) 

只有当帧有效时才运行推理，否则输出全零。

## 4. 后处理（Network_Postprocess） [8](#0-7) 

后处理使用argmax函数选择最大概率的类别（阈值为0.9），然后通过标签过滤器避免输出抖动。 [9](#0-8) 

## Notes

- 该项目使用VL53LMZ多区ToF传感器，可以提供64个区域的距离和信号强度数据
- 数据结构定义在 `app_utils.h` 中，包括 `HANDPOSTURE_Input_Data_t` 和 `HANDPOSTURE_Data_t`
- 预处理参数（MIN_DISTANCE、MAX_DISTANCE、BACKGROUND_REMOVAL）在部署时通过配置文件生成到 `ai_model_config.h` 中
- 整个处理流程确保只有有效的、经过滤波和归一化的数据才会输入到神经网络模型中

### Citations

**File:** application_code/hand_posture/STM32F4/Application/NUCLEO-F401RE/Src/main.c (L76-84)
```c
    {
      /* Wait for available ranging data */
      Sensor_GetRangingData(&App_Config);
      /* Pre-process data */
      Network_Preprocess(&App_Config);
      /* Run inference */
      Network_Inference(&App_Config);
      /* Post-process data */
      Network_Postprocess(&App_Config);
```

**File:** application_code/hand_posture/STM32F4/Application/NUCLEO-F401RE/Src/app_sensor.c (L179-195)
```c
  /* Wait for the sensor to get data */
  if (wait_for_ToF_interrupt(&(App_Config->ToFDev.platform), &(App_Config->IntrCount)) == 0)
  {
    /* Get data from the sensor */
    if (vl53lmz_get_ranging_data(&(App_Config->ToFDev), &(App_Config->RangingData)) != VL53LMZ_STATUS_OK)
    {
      printf("vl53lmz_get_ranging_data failed\n");
      Error_Handler();
    }
    /* Set the flat indicating a new data has been received */
    App_Config->new_data_received = true;
  }
  else
  {
    /* Reset the flag indicating a new data has been received */
    App_Config->new_data_received = false;
  }
```

**File:** application_code/hand_posture/STM32F4/Application/NUCLEO-F401RE/Src/app_network.c (L99-122)
```c
static int label_filter(int current_label, HANDPOSTURE_Data_t *AI_Data)
{
  if (current_label == AI_Data->previous_label)
  {
    if (AI_Data->label_count < LABEL_FILTER_N)
      AI_Data->label_count++;
    else if (AI_Data->label_count == LABEL_FILTER_N)
      AI_Data->handposture_label = current_label;
    else
      AI_Data->label_count = 0;
  }
  else
  {
    AI_Data->label_count = 0;
#if KEEP_LAST_VALID == 0
    /* This line to reset the valid Posture if a different posture is detected,
    by removing this line, we save the previous valid posture until a new valid one is detected */
    AI_Data->handposture_label = 0;
#endif
  }

  AI_Data->previous_label = current_label;
  return(0);
}
```

**File:** application_code/hand_posture/STM32F4/Application/NUCLEO-F401RE/Src/app_network.c (L193-213)
```c
static int AI_CopyInputData(HANDPOSTURE_Input_Data_t *HANDPOSTURE_Input_Data, VL53LMZ_ResultsData *pRangingData)
{
  int idx;

  HANDPOSTURE_Input_Data->timestamp_ms = (int32_t) HAL_GetTick();
  for (int i = 0; i < SENSOR__MAX_NB_OF_ZONES; i++)
  {
    /* Use SENSOR_ROTATION_180 macro to rotate the data */
    #if SENSOR_ROTATION_180
      idx = SENSOR__MAX_NB_OF_ZONES - i;
    #else
      idx = i;
    #endif
    HANDPOSTURE_Input_Data->ranging[idx] = pRangingData->distance_mm[idx]/FIXED_POINT_14_2_TO_FLOAT; /* Signed 14.2 */
    HANDPOSTURE_Input_Data->peak[idx] = pRangingData->signal_per_spad[idx]/FIXED_POINT_21_11_TO_FLOAT; /* Unsigned 21.11 */
    HANDPOSTURE_Input_Data->target_status[idx] = pRangingData->target_status[idx];
    HANDPOSTURE_Input_Data->nb_targets[idx] = pRangingData->nb_target_detected[idx];
  }

  return(0);
}
```

**File:** application_code/hand_posture/STM32F4/Application/NUCLEO-F401RE/Src/app_network.c (L221-257)
```c
static int ValidateFrame(HANDPOSTURE_Data_t *AI_Data, HANDPOSTURE_Input_Data_t *Input_AI_Data)
{
  bool valid;
  int idx;
  float min = 4000.0;

  /* Find minimum valid distance */
  for (idx = 0; idx < SENSOR__MAX_NB_OF_ZONES; idx++){
    if ((Input_AI_Data->nb_targets[idx] > 0)
      && (Input_AI_Data->target_status[idx] == RANGING_OK_5 || Input_AI_Data->target_status[idx] == RANGING_OK_9)
      && Input_AI_Data->ranging[idx] < min)
    {
      min = Input_AI_Data->ranging[idx];
    }
  }

  if (min < MAX_DISTANCE && min > MIN_DISTANCE)
    AI_Data->is_valid_frame = 1;
  else
    AI_Data->is_valid_frame = 0;

  for (idx = 0; idx <SENSOR__MAX_NB_OF_ZONES; idx++)
  {
    /* Check if the data is valid */
    valid = (Input_AI_Data->nb_targets[idx] > 0)
        && (Input_AI_Data->target_status[idx] == RANGING_OK_5 || Input_AI_Data->target_status[idx] == RANGING_OK_9)
        && (Input_AI_Data->ranging[idx] < min + BACKGROUND_REMOVAL);

    /* If not valid, load default value */
    if (!valid)
    {
      Input_AI_Data->ranging[idx] = DEFAULT_RANGING_VALUE;
      Input_AI_Data->peak[idx] = DEFAULT_SIGNAL_VALUE;
    }
  }
  return(0);
}
```

**File:** application_code/hand_posture/STM32F4/Application/NUCLEO-F401RE/Src/app_network.c (L265-277)
```c
static int NormalizeData(float *normalized_data, HANDPOSTURE_Input_Data_t *Input_AI_Data)
{
  int idx;
  for (idx = 0; idx <SENSOR__MAX_NB_OF_ZONES; idx++)
  {
    /* Signed 14.2 */
    normalized_data[2*idx] = (Input_AI_Data->ranging[idx] - NORMALIZATION_RANGING_CENTER) / NORMALIZATION_RANGING_IQR;
    /* Unsigned 21.11 */
    normalized_data[2*idx + 1] = (Input_AI_Data->peak[idx] - NORMALIZATION_SIGNAL_CENTER) / NORMALIZATION_SIGNAL_IQR;
  }

  return(0);
}
```

**File:** application_code/hand_posture/STM32F4/Application/NUCLEO-F401RE/Src/app_network.c (L286-306)
```c
static int output_selection(const float *data, uint32_t len, HANDPOSTURE_Data_t *AI_Data)
{
  int current_label = 0;

  /* If the frame is valid, get the chosen label out of the NN output */
  if (AI_Data->is_valid_frame)
  {
    /* In this example we are using an ArgMax function, but another function can be developed */
    current_label = argmax(data, len, THRESHOLD_NN_OUTPUT);
  }
  /* If the frame is not valid, set the output label as 0 */
  else
  {
    current_label = 0;
  }

  /* Filtering */
  label_filter(current_label, AI_Data);

  return(0);
}
```

**File:** application_code/hand_posture/STM32F4/Application/NUCLEO-F401RE/Src/app_network.c (L369-385)
```c
void Network_Inference(AppConfig_TypeDef *App_Config)
{
  /* If a new data need to be processed and the frame is valid */
  if (App_Config->new_data_received && App_Config->AI_Data.is_valid_frame)
  {
    /* Run NN inference */
    if (AI_Run(App_Config->aiInData, App_Config->aiOutData) < 0)
    {
      printf("AI_Run failed\n");
      Error_Handler();
    }
  }
  else
  {
	  for (int i = 0; i<AI_NETWORK_OUT_1_SIZE; i++) App_Config->aiOutData[i] = 0;
  }

```

**File:** application_code/hand_posture/STM32F4/Application/NUCLEO-F401RE/Inc/app_utils.h (L40-47)
```text
/* Median */
#define NORMALIZATION_RANGING_CENTER              (295)
/* Interquartile range */
#define NORMALIZATION_RANGING_IQR                 (196)
/* Median */
#define NORMALIZATION_SIGNAL_CENTER               (281)
/* Interquartile range */
#define NORMALIZATION_SIGNAL_IQR                  (452)
```
