#!/usr/bin/env bash
# ==============================================================================
# ML Challenge 2026: V11 GPU Cross-Encoder Turnkey Execution Script
# Recommended Instance: AWS EC2 g6e.2xlarge (NVIDIA L40S 48GB / 8 vCPU / 64GB RAM)
# ==============================================================================

set -euo pipefail

echo "================================================================================"
echo "  STARTING V11 GPU CROSS-ENCODER ENVIRONMENT SETUP"
echo "================================================================================"

# 1. Verify NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Ensure NVIDIA drivers are installed on this instance."
    exit 1
fi

nvidia-smi

# 2. Check Python & PyTorch CUDA support
python3 -c "
import torch
print('PyTorch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device Name:', torch.cuda.get_device_name(0))
    print('VRAM (GB):', round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))
    print('BF16 Supported:', torch.cuda.is_bf16_supported())
"

# 3. Install GPU dependencies if needed
echo "Installing/Verifying Python dependencies..."
pip install -q -r requirements_gpu.txt

# 4. Verify test suite
echo "Running test suite..."
python3 -m pytest tests/ -q

# 5. Launch V11 Execution
echo "================================================================================"
echo "  LAUNCHING V11 CROSS-ENCODER PIPELINE"
echo "================================================================================"

# Execute with bf16 mixed precision on L40S, logging output
python3 -u experiments/v11_crossencoder.py \
    --device cuda \
    --target both \
    --bf16 \
    --batch-size 64 \
    --max-length 192 \
    2>&1 | tee logs_v11_execution.log

echo "================================================================================"
echo "  V11 PIPELINE FINISHED. Check logs_v11_execution.log and experiments/v11_results.json"
echo "================================================================================"
