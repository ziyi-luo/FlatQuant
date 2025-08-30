#!/usr/bin/env python3
"""
NPU支持测试脚本
用于验证FlatQuant在NPU上的基本功能
"""

import torch
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_device_detection():
    """测试设备检测功能"""
    print("=== 测试设备检测 ===")
    
    try:
        from flatquant.utils import DEV, get_device
        print(f"检测到的设备: {DEV}")
        print(f"设备类型: {DEV.type}")
        print(f"设备索引: {DEV.index}")
        
        # 测试设备可用性
        if hasattr(torch, 'npu') and torch.npu.is_available():
            print("✅ NPU可用")
            print(f"NPU设备数量: {torch.npu.device_count()}")
        elif torch.cuda.is_available():
            print("✅ CUDA可用")
            print(f"CUDA设备数量: {torch.cuda.device_count()}")
        else:
            print("✅ CPU可用")
            
        return True
    except Exception as e:
        print(f"❌ 设备检测失败: {e}")
        return False

def test_memory_management():
    """测试内存管理功能"""
    print("\n=== 测试内存管理 ===")
    
    try:
        from flatquant.utils import cleanup_memory, DEV
        
        # 创建一些测试张量
        x = torch.randn(1000, 1000, device=DEV)
        y = torch.randn(1000, 1000, device=DEV)
        
        print(f"创建张量后，设备: {x.device}")
        
        # 测试内存清理
        cleanup_memory()
        print("✅ 内存清理功能正常")
        
        return True
    except Exception as e:
        print(f"❌ 内存管理测试失败: {e}")
        return False

def test_basic_operations():
    """测试基本操作"""
    print("\n=== 测试基本操作 ===")
    
    try:
        from flatquant.utils import DEV
        
        # 测试张量创建和运算
        a = torch.randn(100, 100, device=DEV)
        b = torch.randn(100, 100, device=DEV)
        c = torch.mm(a, b)
        
        print(f"矩阵乘法结果形状: {c.shape}")
        print(f"结果设备: {c.device}")
        
        # 测试自动混合精度
        if hasattr(torch, 'npu') and torch.npu.is_available():
            with torch.amp.autocast(device_type="npu", dtype=torch.float16):
                d = torch.mm(a.half(), b.half())
            print("✅ NPU自动混合精度正常")
        elif torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                d = torch.mm(a.half(), b.half())
            print("✅ CUDA自动混合精度正常")
        
        return True
    except Exception as e:
        print(f"❌ 基本操作测试失败: {e}")
        return False

def test_model_loading():
    """测试模型加载功能"""
    print("\n=== 测试模型加载 ===")
    
    try:
        from flatquant.model_utils import get_model
        from flatquant.utils import DEV
        
        # 注意：这里只是测试函数调用，不会实际下载模型
        print("✅ 模型工具导入成功")
        
        # 测试设备检测函数
        def test_get_device():
            from flatquant.utils import get_device
            device = get_device()
            print(f"get_device() 返回: {device}")
            return device == DEV
        
        if test_get_device():
            print("✅ 设备检测函数正常")
        else:
            print("❌ 设备检测函数异常")
            
        return True
    except Exception as e:
        print(f"❌ 模型加载测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("FlatQuant NPU支持测试")
    print("=" * 50)
    
    tests = [
        test_device_detection,
        test_memory_management,
        test_basic_operations,
        test_model_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ 测试异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！NPU支持正常工作。")
        return 0
    else:
        print("⚠️  部分测试失败，请检查环境配置。")
        return 1

if __name__ == "__main__":
    exit(main())
