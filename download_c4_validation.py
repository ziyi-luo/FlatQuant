import os
from datasets import load_dataset

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
output_dir = "/data/disk1/FlatQuant-main/datasets/allenai/c4/"
os.makedirs(output_dir, exist_ok=True)

# 指定只加载验证集对应的 .json.gz 文件
data_files = {
    "validation": "en/c4-validation.*.json.gz"
}

validation_dataset = load_dataset(
    "allenai/c4", 
    data_files=data_files, 
    split="validation"
)

print(validation_dataset)

print("下载完成。开始保存到磁盘...")

# 使用 save_to_disk 方法将数据集保存到指定路径
validation_dataset.save_to_disk(output_dir)

print(f"数据集已成功保存到: {os.path.abspath(output_dir)}")