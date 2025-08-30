#!/usr/bin/env python3
"""
FlatQuant 数据集验证脚本
验证已下载的数据集是否正确
"""

import os
import json
from pathlib import Path
from datasets import load_dataset
import logging

def setup_logging():
    """设置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# 数据集配置
DATASETS_CONFIG = {
    "calibration_datasets": {
        "wikitext": {
            "name": "wikitext",
            "config": "wikitext-2-raw-v1",
            "local_dir": "./datasets/wikitext",
            "expected_splits": ["train", "validation"]
        },
        "c4": {
            "name": "allenai/c4",
            "config": "en",
            "local_dir": "./datasets/allenai/c4",
            "expected_splits": ["train", "validation"]
        },
        "pile": {
            "name": "mit-han-lab/pile-val-backup",
            "config": None,
            "local_dir": "./datasets/pile-val-backup",
            "expected_splits": ["validation"]
        }
    },
    "qa_datasets": {
        "ai2_arc": {
            "name": "allenai/ai2_arc",
            "config": None,
            "local_dir": "./datasets/ai2_arc",
            "expected_splits": ["train", "validation", "test"]
        },
        "hellaswag": {
            "name": "Rowan/hellaswag",
            "config": None,
            "local_dir": "./datasets/hellaswag",
            "expected_splits": ["train", "validation"]
        },
        "lambada_openai": {
            "name": "EleutherAI/lambada_openai",
            "config": None,
            "local_dir": "./datasets/lambada_openai",
            "expected_splits": ["train", "validation", "test"]
        },
        "piqa": {
            "name": "ybisk/piqa",
            "config": None,
            "local_dir": "./datasets/piqa",
            "expected_splits": ["train", "validation"]
        },
        "winogrande": {
            "name": "winogrande",
            "config": "winogrande_xl",
            "local_dir": "./datasets/winogrande",
            "expected_splits": ["train", "validation"]
        }
    }
}

def verify_dataset(dataset_info, dataset_type):
    """验证单个数据集"""
    try:
        logger.info(f"验证 {dataset_type} 数据集: {dataset_info['name']}")
        
        # 检查目录是否存在
        if not Path(dataset_info['local_dir']).exists():
            logger.error(f"❌ 数据集目录不存在: {dataset_info['local_dir']}")
            return False
        
        # 尝试加载数据集
        try:
            dataset = load_dataset(dataset_info['local_dir'])
        except Exception as e:
            logger.error(f"❌ 无法加载数据集: {str(e)}")
            return False
        
        # 检查数据集分割
        available_splits = list(dataset.keys())
        expected_splits = dataset_info['expected_splits']
        
        missing_splits = set(expected_splits) - set(available_splits)
        if missing_splits:
            logger.warning(f"⚠️ 缺少数据集分割: {missing_splits}")
        
        # 显示数据集信息
        logger.info(f"✅ 数据集验证成功: {dataset_info['name']}")
        for split_name in available_splits:
            split_data = dataset[split_name]
            logger.info(f"   - {split_name}: {len(split_data)} 样本")
            if hasattr(split_data, 'features'):
                logger.info(f"     特征: {list(split_data.features.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 验证 {dataset_type} 数据集失败: {dataset_info['name']}")
        logger.error(f"   错误信息: {str(e)}")
        return False

def verify_lm_eval_configs():
    """验证lm_eval配置文件"""
    try:
        logger.info("验证lm_eval配置文件...")
        
        config_dir = Path("./datasets/lm_eval_configs/tasks")
        if not config_dir.exists():
            logger.error(f"❌ lm_eval配置目录不存在: {config_dir}")
            return False
        
        # 检查必要的配置文件
        required_configs = [
            "arc/arc_easy.yaml",
            "arc/arc_challenge.yaml",
            "hellaswag/hellaswag.yaml",
            "lambada/lambada_openai.yaml",
            "piqa/piqa.yaml",
            "winogrande/default.yaml"
        ]
        
        missing_configs = []
        for config_file in required_configs:
            config_path = config_dir / config_file
            if not config_path.exists():
                missing_configs.append(config_file)
            else:
                logger.info(f"✅ 配置文件存在: {config_file}")
        
        if missing_configs:
            logger.warning(f"⚠️ 缺少配置文件: {missing_configs}")
            return False
        
        logger.info("✅ lm_eval配置文件验证成功")
        return True
        
    except Exception as e:
        logger.error(f"❌ lm_eval配置文件验证失败: {str(e)}")
        return False

def check_dataset_info():
    """检查数据集信息文件"""
    try:
        info_file = Path("./datasets/dataset_info.json")
        if not info_file.exists():
            logger.warning("⚠️ 数据集信息文件不存在")
            return False
        
        with open(info_file, 'r', encoding='utf-8') as f:
            info = json.load(f)
        
        logger.info("✅ 数据集信息文件存在")
        logger.info(f"   下载时间: {info.get('download_time', '未知')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 检查数据集信息文件失败: {str(e)}")
        return False

def main():
    """主函数"""
    print("🔍 FlatQuant 数据集验证")
    print("=" * 40)
    
    # 检查数据集信息文件
    check_dataset_info()
    
    # 验证校准数据集
    print("\n📊 验证校准和困惑度评估数据集...")
    calibration_success = 0
    calibration_total = len(DATASETS_CONFIG["calibration_datasets"])
    
    for dataset_name, dataset_info in DATASETS_CONFIG["calibration_datasets"].items():
        if verify_dataset(dataset_info, "校准"):
            calibration_success += 1
    
    # 验证问答数据集
    print("\n❓ 验证常识问答评估数据集...")
    qa_success = 0
    qa_total = len(DATASETS_CONFIG["qa_datasets"])
    
    for dataset_name, dataset_info in DATASETS_CONFIG["qa_datasets"].items():
        if verify_dataset(dataset_info, "问答"):
            qa_success += 1
    
    # 验证lm_eval配置
    print("\n⚙️ 验证lm_eval配置文件...")
    lm_eval_success = verify_lm_eval_configs()
    
    # 显示结果
    print("\n" + "=" * 40)
    print("📋 验证结果汇总:")
    print(f"   校准数据集: {calibration_success}/{calibration_total} 正常")
    print(f"   问答数据集: {qa_success}/{qa_total} 正常")
    print(f"   lm_eval配置: {'✅ 正常' if lm_eval_success else '❌ 异常'}")
    
    total_success = calibration_success + qa_success
    total_datasets = calibration_total + qa_total
    
    if total_success == total_datasets and lm_eval_success:
        print("\n🎉 所有数据集验证通过！")
        print("   现在可以开始使用FlatQuant进行模型量化了")
    else:
        print("\n⚠️ 部分数据集验证失败")
        print("   请运行 ./download_datasets.sh 重新下载数据集")
        
        if calibration_success < calibration_total:
            print("   建议重新下载校准数据集")
        if qa_success < qa_total:
            print("   建议重新下载问答数据集")
        if not lm_eval_success:
            print("   建议重新设置lm_eval配置")

if __name__ == "__main__":
    main()
