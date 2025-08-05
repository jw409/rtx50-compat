# 🚀 Install Qwen with vLLM on RTX 5090

## ⚠️ IMPORTANT: RTX 5090 requires special build process!

The RTX 5090 uses sm_120 architecture which isn't recognized by most packages. You MUST build CUDA extensions targeting sm_90 for compatibility.

## Automated Installation (Recommended)

```bash
# Run our complete installation script
chmod +x install_rtx5090.sh
./install_rtx5090.sh
```

This script handles all the complexity of building for RTX 5090!

## Manual Installation (Advanced Users)

### 1. Create Environment with uv

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate venv
uv venv
source .venv/bin/activate
```

### 2. Set Build Environment (CRITICAL!)

```bash
# Force builds to target sm_90 (H100) for compatibility
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0+PTX"
export CUDA_HOME=/usr/local/cuda
export FORCE_CUDA=1
export MAX_JOBS=4  # Adjust based on CPU cores
```

### 3. Install in Correct Order

```bash
# 1. FIRST install rtx50-compat (enables masquerading)
uv pip install rtx50-compat

# 2. Install PyTorch with CUDA 12.1
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Build xformers from source (5-10 minutes)
uv pip install ninja  # For faster builds
uv pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers

# 4. Build flash-attention (optional but recommended)
uv pip install flash-attn --no-build-isolation

# 5. Install vLLM
uv pip install vllm

# 6. Finally, transformers and other pure Python packages
uv pip install transformers accelerate datasets
```

## Run Qwen Models

### Quick Test

```python
import rtx50_compat  # ALWAYS FIRST!
from vllm import LLM, SamplingParams

# Load Qwen 7B
llm = LLM("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

# Generate
output = llm.generate(["Tell me a joke"], SamplingParams(max_tokens=50))
print(output[0].outputs[0].text)
```

### Available Models for 32GB VRAM

| Model | VRAM Usage | Speed | Use Case |
|-------|------------|-------|----------|
| Qwen2.5-7B | 14-16 GB | 150-200 tok/s | Fast inference |
| Qwen2.5-14B | 28-30 GB | 80-120 tok/s | Balanced |
| Qwen2.5-72B-AWQ | 30-32 GB | 20-30 tok/s | Max quality |
| Qwen2.5-Coder-7B | 14-16 GB | 150-200 tok/s | Code generation |

## Common Issues

### "No kernel image available"
- You forgot to import rtx50_compat first
- You installed packages without setting TORCH_CUDA_ARCH_LIST

### Build fails with "sm_120 not supported"
- You didn't set `export TORCH_CUDA_ARCH_LIST="9.0+PTX"`
- Solution: Clean install with our script

### xformers build takes forever
- Normal! It's compiling CUDA kernels
- Use `export MAX_JOBS=4` to limit CPU usage
- Pre-built wheels don't work for RTX 5090

### "CUDA out of memory"
```python
# Reduce memory usage
llm = LLM(model, gpu_memory_utilization=0.85)
```

## Why This is Complex

1. **RTX 5090 uses sm_120**: Not recognized by PyTorch/xformers
2. **No pre-built wheels**: Everything needs compilation
3. **Binary compatibility**: We target sm_90 and use rtx50-compat to masquerade
4. **Build order matters**: rtx50-compat must be imported first

## Verification

```python
import rtx50_compat
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Masquerading as: {torch.cuda.get_device_capability(0)}")
print(f"xformers available: {'xformers' in dir()}")
```

## Need Help?

- Check our GitHub issues: github.com/jw409/rtx50-compat
- Run the automated script: `./install_rtx5090.sh`
- The community is here to help!

---

**Remember**: Building for RTX 5090 takes time. Get coffee while xformers compiles! ☕