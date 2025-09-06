# FlatQuant 数据集使用指南

本指南说明如何下载和使用FlatQuant所需的数据集。

## 📋 数据集概览

FlatQuant需要两类数据集：

### 1. 校准和困惑度评估数据集
用于模型校准和困惑度评估：
- **WikiText2**: 维基百科文本数据集
- **C4**: Colossal Clean Crawled Corpus
- **Pile**: The Pile验证集

### 2. 常识问答评估数据集
用于常识问答任务评估：
- **AI2 ARC**: AI2推理挑战赛数据集
- **HellaSwag**: 常识推理数据集
- **LAMBADA**: 语言建模基准数据集
- **PIQA**: 物理直觉问答数据集
- **WinoGrande**: 常识推理数据集

## 🚀 快速下载

### 方法1: 使用自动化脚本（推荐）

```bash
cd FlatQuant-main

# 运行下载脚本
./download_datasets.sh
```

### 方法2: 直接使用Python脚本

```bash
cd FlatQuant-main

# 安装依赖（如果需要）
pip install datasets lm-eval

# 运行下载脚本
python download_datasets.py
```

## 📁 目录结构

下载完成后，您将看到以下目录结构：

```
FlatQuant-main/
├── datasets/
│   ├── wikitext/              # WikiText2数据集
│   ├── allenai/
│   │   └── c4/               # C4数据集
│   ├── pile-val-backup/      # Pile验证集
│   ├── ai2_arc/              # AI2 ARC数据集
│   ├── hellaswag/            # HellaSwag数据集
│   ├── lambada_openai/       # LAMBADA数据集
│   ├── piqa/                 # PIQA数据集
│   ├── winogrande/           # WinoGrande数据集
│   ├── lm_eval_configs/
│   │   └── tasks/            # lm_eval配置文件
│   └── dataset_info.json     # 数据集信息文件
```

## 🔧 数据集使用

### 1. 校准数据集使用

FlatQuant会自动使用这些数据集进行模型校准：

```python
# 在量化脚本中，数据集会自动加载
python quantize_qwen2.5_0.5b.py \
    --cali_dataset wikitext2 \
    --nsamples 128
```

### 2. 困惑度评估

```python
# 评估模型在WikiText2上的困惑度
python main.py \
    --model your_model \
    --eval_ppl \
    --eval_datasets wikitext2 c4
```

### 3. 常识问答评估

```python
# 使用lm_eval进行问答任务评估
python main.py \
    --model your_model \
    --lm_eval \
    --tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande
```

## 📊 数据集详情

### WikiText2
- **用途**: 模型校准和困惑度评估
- **大小**: ~2MB
- **格式**: 原始文本
- **配置**: `wikitext-2-raw-v1`

### C4
- **用途**: 模型校准和困惑度评估
- **大小**: ~1GB
- **格式**: 清理后的网页文本
- **配置**: `en`

### AI2 ARC
- **用途**: 常识问答评估
- **大小**: ~15MB
- **格式**: 多选题
- **任务**: `arc_easy`, `arc_challenge`

### HellaSwag
- **用途**: 常识推理评估
- **大小**: ~30MB
- **格式**: 句子补全
- **任务**: `hellaswag`

### LAMBADA
- **用途**: 语言建模评估
- **大小**: ~5MB
- **格式**: 句子补全
- **任务**: `lambada_openai`

### PIQA
- **用途**: 物理直觉评估
- **大小**: ~2MB
- **格式**: 二选一问题
- **任务**: `piqa`

### WinoGrande
- **用途**: 常识推理评估
- **大小**: ~10MB
- **格式**: 代词消解
- **任务**: `winogrande`

## ⚙️ 配置说明

### lm_eval配置文件

脚本会自动复制和修改lm_eval配置文件：

1. **复制配置文件**: 从lm_eval安装目录复制到`./datasets/lm_eval_configs/tasks/`
2. **修改数据集路径**: 将`dataset_path`字段修改为本地路径
3. **支持的任务**: arc_easy, arc_challenge, hellaswag, lambada_openai, piqa, winogrande

### 自定义配置

如果需要使用自定义数据集路径，可以修改配置文件：

```yaml
# 示例: ./datasets/lm_eval_configs/tasks/arc_easy.yaml
dataset_path: ./datasets/ai2_arc
```

## 🐛 常见问题

### 1. 下载失败
```bash
# 检查网络连接
ping huggingface.co

# 重试下载
python download_datasets.py
```

### 2. 内存不足
```bash
# 分批下载数据集
python download_datasets.py --calibration-only  # 只下载校准数据集
python download_datasets.py --qa-only           # 只下载问答数据集
```

### 3. lm_eval配置问题
```bash
# 手动安装lm_eval
pip install lm-eval

# 检查配置文件
ls ./datasets/lm_eval_configs/tasks/
```

### 4. 数据集路径问题
```bash
# 检查数据集是否正确下载
ls -la ./datasets/

# 查看数据集信息
cat ./datasets/dataset_info.json
```

## 📈 性能优化

### 1. 快速测试
对于快速测试，可以使用较小的校准样本数：
```bash
python quantize_qwen2.5_0.5b.py --nsamples 64
```

### 2. 高质量量化
对于生产环境，使用更多校准样本：
```bash
python quantize_qwen2.5_0.5b.py --nsamples 512
```

### 3. 选择性评估
只评估必要的任务：
```bash
python main.py --tasks arc_easy,piqa  # 只评估部分任务
```

## 🔍 验证数据集

下载完成后，可以验证数据集是否正确：

```bash
# 检查数据集文件
python -c "
from datasets import load_dataset
dataset = load_dataset('./datasets/wikitext')
print(f'WikiText2: {len(dataset[\"train\"])} 训练样本, {len(dataset[\"validation\"])} 验证样本')
"
```

## 📞 技术支持

如果遇到问题：

1. 检查日志文件: `dataset_download_*.log`
2. 确认网络连接正常
3. 检查磁盘空间是否充足
4. 参考README.md了解更多信息

## 🎯 最佳实践

1. **首次使用**: 建议下载所有数据集
2. **生产环境**: 根据实际需要选择性下载
3. **存储空间**: 确保有足够的磁盘空间（约2GB）
4. **网络环境**: 确保网络连接稳定
5. **定期更新**: 定期检查数据集更新
