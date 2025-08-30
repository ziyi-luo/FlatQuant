#!/bin/bash

# NPU installation script for FlatQuant
echo "Installing NPU dependencies for FlatQuant..."

# Check if we're on a system with NPU support
if [ -d "/usr/local/Ascend" ] || [ -d "/usr/local/ascend" ]; then
    echo "NPU environment detected"
else
    echo "Warning: NPU environment not detected. This script is for NPU systems."
fi

# Install PyTorch with NPU support
echo "Installing PyTorch with NPU support..."
pip install torch==2.2.1 torch_npu>=2.2.1

# Install other dependencies
echo "Installing other dependencies..."
pip install transformers==4.36.0
pip install accelerate==0.27.2
pip install datasets==2.17.1
pip install lm-eval==0.4.4
pip install termcolor

# Set NPU environment variables
echo "Setting NPU environment variables..."
export ASCEND_DEVICE_ID=0
export ASCEND_VISIBLE_DEVICES=0

echo "NPU installation completed!"
echo "To use NPU, make sure to set the following environment variables:"
echo "export ASCEND_DEVICE_ID=0"
echo "export ASCEND_VISIBLE_DEVICES=0"
