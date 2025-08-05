# RTX 5090 Debug Guide

## Common Issues and Solutions

### Issue 1: PyTorch CUDA Warning
**Symptom**: `NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible`

**Solution**: This is expected! The warning appears but rtx50-compat still makes it work. The masquerading happens at runtime.

### Issue 2: Import Order
**Correct order**:
```python
import rtx50_compat  # MUST be first
import torch         # Then PyTorch
import vllm          # Then other libraries
```

### Issue 3: Environment Setup
**Required for building packages**:
```bash
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0+PTX"
export FORCE_CUDA=1
export CUDA_HOME=/usr/local/cuda
```

### Issue 4: vLLM Installation
```bash
# Full installation sequence
uv pip install rtx50-compat
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install -v git+https://github.com/facebookresearch/xformers.git
uv pip install vllm
```

### Quick Test
```python
import rtx50_compat
import torch
print(torch.cuda.get_device_capability(0))  # Should show (9, 0)
```

## Need Help?
Run: `python -c "import rtx50_compat; rtx50_compat.check_install()"`