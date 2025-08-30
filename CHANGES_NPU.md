# FlatQuant NPU支持修改总结

本文档总结了为FlatQuant添加NPU支持所做的主要修改。

## 修改概述

FlatQuant现在支持在昇腾NPU设备上运行，包括Ascend 910等。代码会自动检测可用的设备，优先级为：NPU > CUDA > CPU。

## 主要修改文件

### 1. 核心设备管理 (`flatquant/utils.py`)
- ✅ 添加了 `get_device()` 函数，优先选择NPU设备
- ✅ 修改了 `cleanup_memory()` 函数以支持NPU内存清理
- ✅ 更新了 `seed_everything()` 函数以支持NPU随机种子设置
- ✅ 修改了 `distribute_model()` 函数以支持NPU设备分发

### 2. 模型工具 (`flatquant/model_tools/`)
- ✅ `llama_utils.py`: 将所有 `.cuda()` 调用替换为 `.to(device)`
- ✅ `llama31_utils.py`: 将所有 `.cuda()` 调用替换为 `.to(device)`
- ✅ `qwen_utils.py`: 将所有 `.cuda()` 调用替换为 `.to(device)`

### 3. 训练工具 (`flatquant/train_utils.py`)
- ✅ 添加了NPU的自动混合精度支持
- ✅ 修改了内存清理逻辑以支持NPU
- ✅ 更新了设备特定的操作

### 4. 主程序文件
- ✅ `main.py`: 使用统一的设备检测
- ✅ `main_dpskv3.py`: 全面支持NPU设备
- ✅ `deepseek_v3/model.py`: 支持NPU设备设置
- ✅ `deepseek_v3/generate.py`: 支持NPU设备操作

### 5. 基准测试 (`benchmarks/`)
- ✅ `qlinear_benchmark.py`: 支持NPU设备
- ✅ `layer_benchmark.py`: 支持NPU设备
- ✅ `qattention_benchmark.py`: 支持NPU设备
- ✅ `kernel_benchmark.py`: 支持NPU设备

### 6. 部署工具 (`deploy/`)
- ✅ `nn/linear.py`: 支持NPU设备
- ✅ `kernels/block_matmul.py`: 支持NPU设备
- ✅ `kernels/kron_matmul.py`: 支持NPU设备

### 7. 其他工具
- ✅ `flatquant/flatness.py`: 支持NPU设备
- ✅ `flatquant/hadamard_utils.py`: 支持NPU设备
- ✅ `gptq_utils.py`: 支持NPU设备

## 新增文件

### 1. 安装脚本 (`install_npu.sh`)
- NPU环境检测
- 自动安装NPU相关依赖
- 环境变量设置

### 2. 使用说明 (`README_NPU.md`)
- 详细的环境要求
- 安装步骤
- 使用方法
- 故障排除

### 3. 测试脚本 (`test_npu.py`)
- 设备检测测试
- 内存管理测试
- 基本操作测试
- 模型加载测试

### 4. 依赖更新 (`requirements.txt`)
- 添加了 `torch_npu>=2.2.1`

## 技术细节

### 设备检测逻辑
```python
def get_device():
    """Get the best available device (NPU > CUDA > CPU)"""
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return torch.device('npu:0')
    elif torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')
```

### 内存管理
- 支持NPU内存清理：`torch.npu.empty_cache()`
- 支持NPU内存监控：`torch.npu.memory_reserved()`
- 自动混合精度支持：`torch.amp.autocast(device_type="npu")`

### 设备设置
- 支持NPU设备设置：`torch.npu.set_device()`
- 支持NPU随机种子：`torch.npu.manual_seed()`
- 支持NPU同步：`torch.npu.synchronize()`

## 兼容性

### 向后兼容
- ✅ 保持与CUDA的完全兼容
- ✅ 保持与CPU的完全兼容
- ✅ 自动检测最佳可用设备

### 环境要求
- PyTorch 2.2.1 + torch_npu
- CANN环境（昇腾计算架构）
- Python 3.8+

## 使用方法

### 基本使用
```bash
# 设置环境变量
export ASCEND_DEVICE_ID=0
export ASCEND_VISIBLE_DEVICES=0

# 运行程序（自动检测NPU）
python main.py --model <model_name> --quantize --w_bits 4 --a_bits 4
```

### 测试NPU支持
```bash
python test_npu.py
```

## 注意事项

1. **性能优化**: NPU版本可能需要根据具体硬件进行进一步优化
2. **内存管理**: NPU的内存管理可能与CUDA有所不同
3. **兼容性**: 某些CUDA特定操作已适配为NPU兼容

## 支持的功能

- ✅ 模型量化（GPTQ、RTN）
- ✅ FlatQuant校准
- ✅ 模型推理
- ✅ 基准测试
- ✅ 内存管理
- ✅ 自动混合精度
- ✅ 分布式训练
- ✅ 模型分发

## 版本信息

- FlatQuant版本: 支持NPU的版本
- PyTorch版本: 2.2.1
- torch_npu版本: >=2.2.1
- 支持的NPU: Ascend 910等
- 修改日期: 2024年

## 贡献者

- 添加了NPU支持
- 保持了向后兼容性
- 提供了完整的文档和测试
