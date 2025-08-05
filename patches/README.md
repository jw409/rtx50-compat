# PyTorch & vLLM Patches for RTX 5090 Support

This directory contains patches for adding native RTX 5090 (sm_120) support to PyTorch and vLLM.

## ⚠️ Important Notes

These patches are provided for reference and transparency. The `rtx50-compat` package provides a runtime solution that doesn't require patching PyTorch or vLLM directly.

**Current Status**: These patches may not apply cleanly to the latest versions of PyTorch/vLLM due to rapid development. They are included to show what changes are needed for native support.

## 📋 Patch Contents

### pytorch_rtx5090.patch
Adds sm_120 support to PyTorch:
- Updates CUDA architecture lists
- Adds RNG offset configuration
- Updates memory management for consumer GPUs
- Adds test coverage for sm_120

### vllm_rtx5090.patch
Adds sm_120 support to vLLM:
- Updates compute capability checks
- Enables FlashAttention for sm_120
- Optimizes memory utilization for 32GB consumer GPUs
- Adds RTX 50-series detection

## 🔧 Applying Patches (Advanced Users)

If you want to build PyTorch/vLLM from source with native RTX 5090 support:

### PyTorch
```bash
git clone https://github.com/pytorch/pytorch.git
cd pytorch
git checkout v2.5.0  # Or latest stable
git apply /path/to/pytorch_rtx5090.patch
python setup.py install
```

### vLLM
```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.5.0  # Or latest stable
git apply /path/to/vllm_rtx5090.patch
pip install -e .
```

## 🚨 Common Issues

1. **Patch doesn't apply cleanly**: The codebase has changed. You'll need to apply changes manually by looking at the patch contents.

2. **Build errors**: Ensure you have CUDA 12.0+ and all build dependencies installed.

3. **Runtime errors**: The runtime patching approach (using the rtx50-compat package) is more reliable than source patches.

## 🎯 Why Runtime Patching?

The `rtx50-compat` package uses runtime patching instead of source patches because:
- No need to rebuild PyTorch/vLLM
- Works with pip-installed packages
- Automatically updates with new PyTorch versions
- Easier for end users

## 📝 Contributing

If you successfully apply these patches to newer versions, please submit a PR with updated patches!