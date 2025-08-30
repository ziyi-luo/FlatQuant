# FlatQuant NPU 支持说明

本文档说明如何在NPU（昇腾）设备上运行FlatQuant。

## 环境要求

- 昇腾NPU设备（如Ascend 910）
- CANN（Compute Architecture for Neural Networks）环境
- Python 3.8+
- PyTorch 2.2.1 + torch_npu

## 安装步骤

### 1. 安装NPU依赖

```bash
# 运行NPU安装脚本
./install_npu.sh

# 或者手动安装
pip install torch==2.2.1 torch_npu>=2.2.1
pip install transformers==4.36.0 accelerate==0.27.2 datasets==2.17.1 lm-eval==0.4.4 termcolor
```

### 2. 设置环境变量

```bash
export ASCEND_DEVICE_ID=0
export ASCEND_VISIBLE_DEVICES=0
```

## 使用方法

### 基本使用

FlatQuant现在会自动检测可用的设备，优先级为：NPU > CUDA > CPU

```bash
# 运行主程序
python main.py --model <model_name> --quantize --w_bits 4 --a_bits 4

# 运行DeepSeek V3版本
python main_dpskv3.py --ckpt-path <path> --config <config> --quantize --w_bits 4 --a_bits 4
```

### 设备检测

代码会自动检测设备类型：
- 如果检测到NPU，将使用 `npu:0` 设备
- 如果检测到CUDA，将使用 `cuda:0` 设备
- 否则使用CPU

### 内存管理

NPU版本包含了针对NPU的内存管理优化：
- 自动清理NPU内存缓存
- 支持NPU设备的内存监控
- 适配NPU的自动混合精度训练

## 主要修改

### 1. 设备检测 (`flatquant/utils.py`)
- 添加了 `get_device()` 函数，优先选择NPU设备
- 修改了内存清理函数以支持NPU
- 更新了随机种子设置以支持NPU

### 2. 模型工具 (`flatquant/model_tools/`)
- 将所有 `.cuda()` 调用替换为 `.to(device)` 以支持NPU
- 修改了设备特定的张量创建

### 3. 训练工具 (`flatquant/train_utils.py`)
- 添加了NPU的自动混合精度支持
- 修改了内存清理逻辑

### 4. 基准测试 (`benchmarks/`)
- 更新了所有基准测试以支持NPU设备
- 修改了同步和内存管理函数

## 注意事项

1. **性能优化**: NPU版本可能需要根据具体的NPU硬件进行进一步优化
2. **内存使用**: NPU的内存管理可能与CUDA有所不同，请根据实际情况调整批次大小
3. **兼容性**: 某些CUDA特定的操作可能需要在NPU上进行适配

## 故障排除

### 常见问题

1. **NPU不可用**
   ```
   检查是否正确安装了torch_npu
   确认环境变量设置正确
   ```

2. **内存不足**
   ```
   减少批次大小
   检查ASCEND_VISIBLE_DEVICES设置
   ```

3. **性能问题**
   ```
   确保使用最新的CANN版本
   检查NPU驱动是否正确安装
   ```

## 支持的功能

- ✅ 模型量化（GPTQ、RTN）
- ✅ FlatQuant校准
- ✅ 模型推理
- ✅ 基准测试
- ✅ 内存管理
- ✅ 自动混合精度

## 版本信息

- FlatQuant版本: 支持NPU的版本
- PyTorch版本: 2.2.1
- torch_npu版本: >=2.2.1
- 支持的NPU: Ascend 910等
