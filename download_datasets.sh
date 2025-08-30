#!/bin/bash

# FlatQuant 数据集下载脚本
# 根据README.md要求下载所有必要的数据集

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo "📊 FlatQuant 数据集下载脚本"
echo "=============================="

# # 检查Python环境
# print_info "检查Python环境..."
# if ! command -v python &> /dev/null; then
#     print_error "Python未找到，请确保Python已安装"
#     exit 1
# fi

# # 检查必要的Python包
# print_info "检查Python依赖..."
# python -c "import datasets" 2>/dev/null || {
#     print_error "缺少datasets包，正在安装..."
#     pip install datasets
# }

# python -c "import lm_eval" 2>/dev/null || {
#     print_warning "缺少lm_eval包，正在安装..."
#     pip install lm-eval
# }

# print_success "Python环境检查通过"

# 显示将要下载的数据集
echo ""
print_info "将要下载的数据集:"
echo "📊 校准和困惑度评估数据集:"
echo "   - WikiText2 (wikitext-2-raw-v1)"
echo "   - C4 (allenai/c4)"
echo "   - Pile (mit-han-lab/pile-val-backup)"
echo ""
echo "❓ 常识问答评估数据集:"
echo "   - AI2 ARC (allenai/ai2_arc)"
echo "   - HellaSwag (Rowan/hellaswag)"
echo "   - LAMBADA (EleutherAI/lambada_openai)"
echo "   - PIQA (ybisk/piqa)"
echo "   - WinoGrande (winogrande)"
echo ""


# 运行Python下载脚本
print_info "开始下载数据集..."

# 显示下载选项
echo ""
print_info "下载选项:"
echo "   - 智能检测: 检查数据集完整性，只下载缺失部分"
echo "   - 断点续传: 支持中断后继续下载"
echo "   - 分割下载: 只下载指定分割（节省空间和时间）"
echo "   - 强制重新下载: 使用 --force 参数"
echo "   - 完整数据集: 使用 --full-datasets 参数"

# 运行下载脚本
if python download_datasets.py --force --qa-only --skip_lm_eval; then
    print_success "数据集下载完成！"
    
#     echo ""
#     print_info "数据集位置:"
#     echo "   📁 校准数据集: ./datasets/"
#     echo "   📁 lm_eval配置: ./datasets/lm_eval_configs/tasks/"
#     echo "   📄 详细信息: ./datasets/dataset_info.json"
    
#     echo ""
#     print_info "使用说明:"
#     echo "   1. 校准数据集可直接用于FlatQuant量化"
#     echo "   2. 问答数据集需要配合lm_eval使用"
#     echo "   3. 请参考README.md了解详细使用方法"
    
#     # 显示数据集大小
#     echo ""
#     print_info "数据集大小统计:"
#     if [ -d "./datasets" ]; then
#         du -sh ./datasets/* 2>/dev/null | while read size path; do
#             echo "   $size $path"
#         done
#     fi
    
#     # 验证数据集
#     echo ""
#     print_info "验证数据集完整性..."
#     if python verify_datasets.py; then
#         print_success "数据集验证通过！"
#     else
#         print_warning "部分数据集验证失败，请检查日志"
#     fi
    
# else
#     print_error "数据集下载失败"
#     print_info "请检查日志文件: dataset_download_*.log"
#     print_info "常见解决方案:"
#     echo "   1. 检查网络连接: ping huggingface.co"
#     echo "   2. 检查磁盘空间: df -h"
#     echo "   3. 强制重新下载: python download_datasets.py --force"
#     echo "   4. 只下载校准数据集: python download_datasets.py --calibration-only"
#     exit 1
fi

echo ""
print_success "🎉 数据集准备完成！现在可以开始使用FlatQuant进行模型量化了"
