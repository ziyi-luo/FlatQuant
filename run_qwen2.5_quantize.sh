#!/bin/bash

# Qwen2.5-0.5B-Instruct 模型量化脚本
# 使用方法: ./run_qwen2.5_quantize.sh [选项]

set -e  # 遇到错误时退出

# 默认参数
MODEL_PATH="../Qwen2.5-0.5B-Instruct"
OUTPUT_DIR="./qwen2.5_0.5b_quantized"
W_BITS=4
A_BITS=4
NSAMPLES=128
EPOCHS=15
CALI_DATASET="wikitext2"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
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

# 显示帮助信息
show_help() {
    echo "Qwen2.5-0.5B-Instruct 模型量化脚本"
    echo ""
    echo "使用方法:"
    echo "  $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -m, --model PATH        模型路径 (默认: $MODEL_PATH)"
    echo "  -o, --output DIR        输出目录 (默认: $OUTPUT_DIR)"
    echo "  -w, --w-bits BITS       权重量化位数 (默认: $W_BITS)"
    echo "  -a, --a-bits BITS       激活量化位数 (默认: $A_BITS)"
    echo "  -n, --nsamples NUM      校准样本数 (默认: $NSAMPLES)"
    echo "  -e, --epochs NUM        训练轮数 (默认: $EPOCHS)"
    echo "  -d, --dataset DATASET   校准数据集 (默认: $CALI_DATASET)"
    echo "  --no-flatquant          禁用FlatQuant"
    echo "  --no-eval               禁用评估"
    echo "  --gptq                  使用GPTQ而不是RTN"
    echo "  -h, --help              显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                                    # 使用默认参数"
    echo "  $0 -w 4 -a 4 -n 256                   # 4位量化，256个校准样本"
    echo "  $0 --no-flatquant --gptq              # 只使用GPTQ量化"
    echo "  $0 -o ./my_quantized_model            # 自定义输出目录"
}

# 解析命令行参数
FLATQUANT=true
EVAL=true
GPTQ=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -w|--w-bits)
            W_BITS="$2"
            shift 2
            ;;
        -a|--a-bits)
            A_BITS="$2"
            shift 2
            ;;
        -n|--nsamples)
            NSAMPLES="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -d|--dataset)
            CALI_DATASET="$2"
            shift 2
            ;;
        --no-flatquant)
            FLATQUANT=false
            shift
            ;;
        --no-eval)
            EVAL=false
            shift
            ;;
        --gptq)
            GPTQ=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查模型路径
if [[ ! -d "$MODEL_PATH" ]]; then
    print_error "模型路径不存在: $MODEL_PATH"
    print_info "请确保Qwen2.5-0.5B-Instruct模型已下载到正确位置"
    exit 1
fi

# 检查必要的文件
if [[ ! -f "$MODEL_PATH/config.json" ]] || [[ ! -f "$MODEL_PATH/model.safetensors" ]]; then
    print_error "模型文件不完整，请检查以下文件是否存在:"
    print_error "  - $MODEL_PATH/config.json"
    print_error "  - $MODEL_PATH/model.safetensors"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

print_info "开始量化 Qwen2.5-0.5B-Instruct 模型"
print_info "模型路径: $MODEL_PATH"
print_info "输出目录: $OUTPUT_DIR"
print_info "权重量化位数: $W_BITS"
print_info "激活量化位数: $A_BITS"
print_info "校准样本数: $NSAMPLES"
print_info "训练轮数: $EPOCHS"
print_info "校准数据集: $CALI_DATASET"
print_info "FlatQuant: $FLATQUANT"
print_info "评估: $EVAL"
print_info "GPTQ: $GPTQ"

# 构建Python命令
PYTHON_CMD="python quantize_qwen2.5_0.5b.py"

# 添加参数
PYTHON_CMD="$PYTHON_CMD --model $MODEL_PATH"
PYTHON_CMD="$PYTHON_CMD --output_dir $OUTPUT_DIR"
PYTHON_CMD="$PYTHON_CMD --w_bits $W_BITS"
PYTHON_CMD="$PYTHON_CMD --a_bits $A_BITS"
PYTHON_CMD="$PYTHON_CMD --nsamples $NSAMPLES"
PYTHON_CMD="$PYTHON_CMD --epochs $EPOCHS"
PYTHON_CMD="$PYTHON_CMD --cali_dataset $CALI_DATASET"

if [[ "$FLATQUANT" == "false" ]]; then
    PYTHON_CMD="$PYTHON_CMD --no_flatquant"
fi

if [[ "$EVAL" == "false" ]]; then
    PYTHON_CMD="$PYTHON_CMD --no_eval"
fi

if [[ "$GPTQ" == "true" ]]; then
    PYTHON_CMD="$PYTHON_CMD --gptq"
fi

print_info "执行命令: $PYTHON_CMD"

# 检查Python环境
if ! command -v python &> /dev/null; then
    print_error "Python未找到，请确保Python已安装"
    exit 1
fi

# 检查必要的Python包
print_info "检查Python依赖..."
python -c "import torch, transformers" 2>/dev/null || {
    print_error "缺少必要的Python包，请安装: pip install torch transformers"
    exit 1
}

# 运行量化
print_info "开始执行量化..."
if eval $PYTHON_CMD; then
    print_success "量化完成！"
    print_success "量化后的模型保存在: $OUTPUT_DIR"
    print_info "您可以使用以下命令加载量化后的模型:"
    echo "from transformers import AutoModelForCausalLM, AutoTokenizer"
    echo "model = AutoModelForCausalLM.from_pretrained('$OUTPUT_DIR/quantized_model')"
    echo "tokenizer = AutoTokenizer.from_pretrained('$OUTPUT_DIR/quantized_model')"
else
    print_error "量化失败，请检查日志文件"
    exit 1
fi
