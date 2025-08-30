#!/usr/bin/env python3
"""
FlatQuant 数据集下载脚本
根据README.md要求下载所有必要的数据集
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datasets import load_dataset
import logging
from datetime import datetime

# 设置日志
def setup_logging():
    """设置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'dataset_download_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# 数据集配置
DATASETS_CONFIG = {
    # 校准和困惑度评估数据集
    "calibration_datasets": {
        # "wikitext": {
        #     "name": "wikitext",
        #     "config": "wikitext-2-raw-v1",
        #     "local_dir": "./datasets/wikitext",
        #     "url": "https://huggingface.co/datasets/wikitext",
        #     "expected_splits": ["train", "validation"]
        # },
        "c4": {
            "name": "allenai/c4",
            "config": "en",
            "local_dir": "./datasets/allenai/c4",
            "url": "https://huggingface.co/datasets/allenai/c4",
            "expected_splits": ["train", "validation"]
        },
        # "pile": {
        #     "name": "mit-han-lab/pile-val-backup",
        #     "config": None,
        #     "local_dir": "./datasets/pile-val-backup",
        #     "url": "https://huggingface.co/datasets/mit-han-lab/pile-val-backup",
        #     "expected_splits": ["validation"]
        # }
    },
    
    # 常识问答评估数据集
    "qa_datasets": {
        "ai2_arc": {
            "name": "allenai/ai2_arc",
            "config": ["ARC-Challenge", "ARC-Easy"],
            "local_dir": "./datasets/ai2_arc",
            "url": "https://huggingface.co/datasets/allenai/ai2_arc",
            "expected_splits": ["train", "validation", "test"]
        },
        "hellaswag": {
            "name": "Rowan/hellaswag",
            "config": None,
            "local_dir": "./datasets/hellaswag",
            "url": "https://huggingface.co/datasets/Rowan/hellaswag",
            "expected_splits": ["train", "validation"]
        },
        "lambada_openai": {
            "name": "EleutherAI/lambada_openai",
            "config": None,
            "local_dir": "./datasets/lambada_openai",
            "url": "https://huggingface.co/datasets/EleutherAI/lambada_openai",
            "expected_splits": ["test"]
        },
        "piqa": {
            "name": "ybisk/piqa",
            "config": None,
            "local_dir": "./datasets/piqa",
            "url": "https://huggingface.co/datasets/ybisk/piqa",
            "expected_splits": ["train", "validation"]
        },
        "winogrande": {
            "name": "winogrande",
            "config": "winogrande_xl",
            "local_dir": "./datasets/winogrande",
            "url": "https://huggingface.co/datasets/winogrande",
            "expected_splits": ["train", "validation"]
        }
    }
}

def create_directories():
    """创建必要的目录结构"""
    directories = [
        "./datasets",
        "./datasets/wikitext",
        "./datasets/allenai/c4",
        "./datasets/pile-val-backup",
        "./datasets/ai2_arc",
        "./datasets/hellaswag",
        "./datasets/lambada_openai",
        "./datasets/piqa",
        "./datasets/winogrande",
        "./datasets/lm_eval_configs",
        "./datasets/lm_eval_configs/tasks"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"创建目录: {directory}")

# def check_dataset_completeness(dataset_info):
#     """检查数据集是否完整下载"""
#     try:
#         dataset_dir = Path(dataset_info['local_dir'])
        
#         # 如果目录不存在，说明未开始下载
#         if not dataset_dir.exists():
#             return "not_started"
        
#         # 检查是否有数据集文件
#         dataset_files = list(dataset_dir.rglob("*.arrow")) + list(dataset_dir.rglob("*.parquet")) + list(dataset_dir.rglob("dataset_info.json"))
        
#         if not dataset_files:
#             return "incomplete"
        
#         # 尝试加载数据集验证完整性
#         try:
#             dataset = load_dataset(dataset_info['local_dir'])
            
#             # 检查是否有数据
#             total_samples = 0
#             for split_name, split_data in dataset.items():
#                 total_samples += len(split_data)
            
#             if total_samples == 0:
#                 return "incomplete"
            
#             # 检查预期的分割是否存在
#             expected_splits = dataset_info.get('expected_splits', [])
#             if expected_splits:
#                 available_splits = list(dataset.keys())
#                 missing_splits = set(expected_splits) - set(available_splits)
#                 if missing_splits:
#                     logger.warning(f"数据集缺少分割: {missing_splits}")
#                     return "incomplete"
            
#             return "complete"
            
#         except Exception as e:
#             logger.warning(f"数据集加载失败，可能不完整: {str(e)}")
#             return "incomplete"
            
#     except Exception as e:
#         logger.warning(f"检查数据集完整性时出错: {str(e)}")
#         return "incomplete"

def check_specific_config_completeness(dataset_info, config):
    """检查特定配置的子数据集是否完整"""
    try:
        dataset_dir = Path(dataset_info['local_dir'])
        
        # 构建特定配置的子目录路径
        if config:
            config_dir = dataset_dir / config
            if not config_dir.exists():
                return False
        else:
            config_dir = dataset_dir
        
        # 检查该配置下是否有数据文件
        data_files = list(config_dir.rglob("*.arrow")) + list(config_dir.rglob("*.parquet"))
        if not data_files:
            return False
        
        # 尝试加载该配置的数据集
        load_kwargs = {}
        if config:
            load_kwargs["name"] = config
            
        dataset = load_dataset(
            dataset_info['name'],
            cache_dir=str(dataset_dir),** load_kwargs
        )
        
        # 检查是否有数据
        total_samples = 0
        for split_name, split_data in dataset.items():
            total_samples += len(split_data)
        
        if total_samples == 0:
            return False
        
        # 检查预期的分割是否存在
        expected_splits = dataset_info.get('expected_splits', [])
        if expected_splits:
            available_splits = list(dataset.keys())
            missing_splits = set(expected_splits) - set(available_splits)
            if missing_splits:
                logger.warning(f"配置 {config} 缺少分割: {missing_splits}")
                return False
        
        return True
        
    except Exception as e:
        logger.warning(f"检查配置 {config} 完整性时出错: {str(e)}")
        return False

def download_dataset(dataset_info, dataset_type, force=False):
    """下载单个数据集（支持多配置）"""
    try:
        logger.info(f"开始处理 {dataset_type} 数据集: {dataset_info['name']}")
        
        # 确保config始终是可迭代对象，修复变量未赋值问题
        config_value = dataset_info.get('config')
        if isinstance(config_value, list):
            configs = config_value
        elif config_value is not None:
            configs = [config_value]
        else:
            configs = [None]  # 明确设置为包含None的列表，确保config有值
        
        all_success = True
        
        for config in configs:
            # 检查当前配置是否已完整下载
            is_complete = check_specific_config_completeness(dataset_info, config)
            
            # 强制重新下载或不完整时才下载
            if force or not is_complete:
                if force and Path(dataset_info['local_dir']).exists():
                    # 只删除当前配置的目录（如果存在）
                    if config:
                        config_dir = Path(dataset_info['local_dir']) / config
                        if config_dir.exists():
                            logger.info(f"🔄 强制删除配置 {config} 的现有数据")
                            shutil.rmtree(config_dir)
                
                logger.info(f"📥 开始下载配置 {config or 'default'}")
                
                # 下载当前配置
                download_kwargs = {"trust_remote_code": True}
                if config:
                    download_kwargs["name"] = config
                
                try:
                    dataset = load_dataset(
                        dataset_info['name'],
                        cache_dir=dataset_info['local_dir'],
                        **download_kwargs
                    )
                    
                    logger.info(f"✅ 成功下载配置 {config or 'default'}")
                    for split_name, split_data in dataset.items():
                        logger.info(f"   - {split_name}: {len(split_data)} 样本")
                        
                except Exception as e:
                    logger.error(f"❌ 下载配置 {config or 'default'} 失败: {str(e)}")
                    all_success = False
            else:
                logger.info(f"✅ 配置 {config or 'default'} 已完整，跳过下载")
        
        return all_success
        
    except Exception as e:
        logger.error(f"❌ 处理数据集 {dataset_info['name']} 时出错: {str(e)}")
        return False

def setup_lm_eval_configs():
    """设置lm_eval配置文件"""
    try:
        logger.info("设置lm_eval配置文件...")
        
        # 查找lm_eval安装路径
        import lm_eval
        lm_eval_path = Path(lm_eval.__file__).parent / "tasks"
        
        if not lm_eval_path.exists():
            logger.warning(f"未找到lm_eval tasks目录: {lm_eval_path}")
            return False
        
        # 复制配置文件
        target_path = Path("/data/disk1/FlatQuant-main/datasets/lm_eval_configs/tasks")
        target_path.mkdir(parents=True, exist_ok=True)
        
        # 需要复制的配置文件
        config_files = [
            "arc/arc_easy.yaml",
            "arc/arc_challenge.yaml", 
            "hellaswag/hellaswag.yaml",
            "lambada/lambada_openai.yaml",
            "piqa/piqa.yaml",
            "winogrande/default.yaml"
        ]

        copied_count = 0
        for config_file in config_files:
            source_file = lm_eval_path / config_file
            target_file = target_path / config_file
            print("source_file", source_file)
            print("target_file", target_file)
            target_file.parent.mkdir(parents=True, exist_ok=True)
            if source_file.exists():
                shutil.copy2(source_file, target_file)
                logger.info(f"复制配置文件: {config_file}")
                copied_count += 1
            else:
                logger.warning(f"配置文件不存在: {source_file}")
        
        if copied_count > 0:
            logger.info(f"✅ 成功复制 {copied_count} 个lm_eval配置文件")
            
            # 修改配置文件中的数据集路径
            modify_lm_eval_configs()
            return True
        else:
            logger.warning("未找到任何lm_eval配置文件")
            return False
            
    except Exception as e:
        logger.error(f"❌ 设置lm_eval配置文件失败: {str(e)}")
        return False

def modify_lm_eval_configs():
    """修改lm_eval配置文件中的数据集路径"""
    config_mappings = {
        "arc_easy.yaml": "./datasets/ai2_arc",
        "arc_challenge.yaml": "./datasets/ai2_arc", 
        "hellaswag.yaml": "./datasets/hellaswag",
        "lambada_openai.yaml": "./datasets/lambada_openai",
        "piqa.yaml": "./datasets/piqa",
        "winogrande.yaml": "./datasets/winogrande"
    }
    
    config_dir = Path("./datasets/lm_eval_configs/tasks")
    
    for config_file, dataset_path in config_mappings.items():
        config_path = config_dir / config_file
        if config_path.exists():
            try:
                # 读取配置文件
                with open(config_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 修改数据集路径
                # 这里需要根据具体的配置文件格式进行调整
                # 通常需要将dataset_path字段修改为本地路径
                modified_content = content.replace(
                    "dataset_path: null",
                    f"dataset_path: {dataset_path}"
                )
                
                # 写回文件
                with open(config_path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
                
                logger.info(f"修改配置文件: {config_file}")
                
            except Exception as e:
                logger.warning(f"修改配置文件失败 {config_file}: {str(e)}")

def create_dataset_info_file():
    """创建数据集信息文件"""
    info = {
        "download_time": datetime.now().isoformat(),
        "datasets": DATASETS_CONFIG,
        "usage": {
            "calibration_datasets": "用于模型校准和困惑度评估",
            "qa_datasets": "用于常识问答任务评估"
        },
        "notes": [
            "所有数据集已下载到 ./datasets/ 目录",
            "lm_eval配置文件已复制到 ./datasets/lm_eval_configs/tasks/",
            "请确保在使用时设置正确的数据集路径"
        ]
    }
    
    with open("./datasets/dataset_info.json", 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    logger.info("创建数据集信息文件: ./datasets/dataset_info.json")

def main():
    """主函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='下载FlatQuant所需的数据集')
    parser.add_argument('--calibration-only', action='store_true', 
                        help='只下载校准数据集')
    parser.add_argument('--qa-only', action='store_true', 
                        help='只下载问答数据集')
    parser.add_argument('--skip_lm_eval', action='store_true', 
                        help='跳过lm_eval配置设置')
    parser.add_argument('--force', action='store_true', 
                        help='强制重新下载已存在的数据集')
    
    args = parser.parse_args()
    
    print("🚀 开始下载FlatQuant所需的数据集")
    print("=" * 60)
    
    # 创建目录结构
    create_directories()
    
    # 下载校准数据集
    if not args.qa_only:
        print("\n📊 下载校准和困惑度评估数据集...")
        calibration_success = 0
        calibration_total = len(DATASETS_CONFIG["calibration_datasets"])
        
        for dataset_name, dataset_info in DATASETS_CONFIG["calibration_datasets"].items():
            if download_dataset(dataset_info, "校准", force=args.force):
                calibration_success += 1
    else:
        calibration_success = 0
        calibration_total = 0
    
    # 下载问答数据集
    if not args.calibration_only:
        print("\n❓ 下载常识问答评估数据集...")
        qa_success = 0
        qa_total = len(DATASETS_CONFIG["qa_datasets"])
        
        for dataset_name, dataset_info in DATASETS_CONFIG["qa_datasets"].items():
            if download_dataset(dataset_info, "问答", force=args.force):
                qa_success += 1
    else:
        qa_success = 0
        qa_total = 0
    
    # 设置lm_eval配置
    lm_eval_success = False
    if not args.skip_lm_eval:
        print("\n⚙️ 设置lm_eval配置文件...")
        lm_eval_success = setup_lm_eval_configs()
    
    # 创建信息文件
    create_dataset_info_file()
    
    # 显示结果
    print("\n" + "=" * 60)
    print("📋 下载结果汇总:")
    if calibration_total > 0:
        print(f"   校准数据集: {calibration_success}/{calibration_total} 成功")
    if qa_total > 0:
        print(f"   问答数据集: {qa_success}/{qa_total} 成功")
    if not args.skip_lm_eval:
        print(f"   lm_eval配置: {'✅ 成功' if lm_eval_success else '❌ 失败'}")
    
    total_success = calibration_success + qa_success
    total_datasets = calibration_total + qa_total
    
    if total_datasets == 0:
        print("\n⚠️ 没有选择要下载的数据集")
    elif total_success == total_datasets:
        print("\n🎉 所有数据集下载完成！")
        print("\n📁 数据集位置:")
        print("   - 校准数据集: ./datasets/")
        print("   - lm_eval配置: ./datasets/lm_eval_configs/tasks/")
        print("   - 详细信息: ./datasets/dataset_info.json")
        
        print("\n🔧 使用说明:")
        print("   1. 校准数据集可直接用于FlatQuant量化")
        print("   2. 问答数据集需要配合lm_eval使用")
        print("   3. 请参考README.md了解详细使用方法")
    else:
        print("\n⚠️ 部分数据集下载失败，请检查网络连接和错误日志")
        print("   您可以稍后重新运行此脚本下载失败的数据集")

if __name__ == "__main__":
    main()
