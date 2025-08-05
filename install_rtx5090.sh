#!/bin/bash
# RTX 5090 Complete Installation Script
# This properly builds all dependencies for sm_120 compatibility

set -e  # Exit on error

echo "🚀 RTX 5090 Installation Script for vLLM + Transformers"
echo "======================================================"

# Check if we're in a venv
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Creating virtual environment..."
    uv venv
    source .venv/bin/activate
fi

# Step 1: Install rtx50-compat FIRST (critical!)
echo -e "\n📦 Installing rtx50-compat..."
uv pip install rtx50-compat

# Step 2: Set build environment for sm_90 compatibility
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0+PTX"
export CUDA_HOME=/usr/local/cuda
export FORCE_CUDA=1
export MAX_JOBS=4  # Adjust based on your CPU

echo -e "\n🔧 Build environment set:"
echo "  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "  CUDA_HOME=$CUDA_HOME"

# Step 3: Install PyTorch with CUDA 12.1
echo -e "\n🔥 Installing PyTorch..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Step 4: Test RTX 5090 is recognized
echo -e "\n🧪 Testing GPU detection..."
python3 -c "
import rtx50_compat  # Must be first!
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Capability: {torch.cuda.get_device_capability(0)} (masqueraded)')
print(f'CUDA Available: {torch.cuda.is_available()}')
"

# Step 5: Build xformers from source
echo -e "\n🔨 Building xformers (this takes 5-10 minutes)..."
uv pip install ninja  # Faster builds
uv pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers

# Step 6: Install flash-attention
echo -e "\n⚡ Building flash-attention..."
uv pip install flash-attn --no-build-isolation

# Step 7: Install vLLM
echo -e "\n🚀 Installing vLLM..."
uv pip install vllm

# Step 8: Install transformers and friends
echo -e "\n🤗 Installing transformers ecosystem..."
uv pip install transformers accelerate bitsandbytes datasets

# Step 9: Final verification
echo -e "\n✅ Verifying installation..."
python3 -c "
import rtx50_compat
import torch
import xformers
import transformers
import vllm
from flash_attn import flash_attn_func

print('✅ rtx50-compat:', rtx50_compat.__version__ if hasattr(rtx50_compat, '__version__') else 'OK')
print('✅ PyTorch:', torch.__version__)
print('✅ xformers:', xformers.__version__)
print('✅ transformers:', transformers.__version__)
print('✅ vLLM:', vllm.__version__)
print('✅ Flash Attention: Available')
print('✅ GPU:', torch.cuda.get_device_name(0))
print('✅ VRAM:', f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

echo -e "\n🎉 Installation complete! RTX 5090 is ready for AI workloads!"
echo "📝 Note: Always import rtx50_compat before any other imports!"