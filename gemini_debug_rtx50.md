# Debug rtx50-compat v2.0.0 with Gemini

## Context
I've just published rtx50-compat v2.0.0 to PyPI with AI-assisted self-healing installation features. Need your help to verify it works correctly.

## Package Details
- **PyPI**: https://pypi.org/project/rtx50-compat/
- **Version**: 2.0.0 (just published)
- **Key Feature**: AI-assisted installation that detects claude/gemini CLI and helps users fix build issues

## What's New in v2.0.0
1. **AIAssistant class**: Detects available AI CLIs (claude/gemini)
2. **SmartInstaller class**: Checks environment and auto-generates fixes
3. **get_ai_help_for_error()**: Gets AI help for specific errors
4. **Auto-import hooks**: Catches xformers/flash_attn import errors and provides AI help

## Core Functionality
The package makes RTX 5090 (sm_120) masquerade as H100 (sm_90) for PyTorch compatibility:
```python
import rtx50_compat  # Must be first!
import torch
# Now RTX 5090 works with all CUDA libraries
```

## Test Instructions
Please help me verify the package works:

### 1. Install and Basic Test
```bash
# Create test environment
uv venv test_rtx50
source test_rtx50/bin/activate

# Install package
uv pip install rtx50-compat

# Test import
python -c "import rtx50_compat; print(f'Version: {rtx50_compat.__version__}')"
```

### 2. Test AI Detection
```python
from rtx50_compat import AIAssistant, SmartInstaller

# Check if AI assistants are detected
assistants = AIAssistant.detect_assistants()
print(f"AI assistants found: {assistants}")
# Should show: {'claude': True, 'gemini': True} if both installed

# Test SmartInstaller
installer = SmartInstaller()
issues = installer.check_environment()
print(f"Environment issues: {issues}")
```

### 3. Test Error Help
```python
from rtx50_compat import get_ai_help_for_error

# Simulate a common error
error = "ImportError: libcudnn_ops_infer.so.8: cannot open shared object file"
help_text = get_ai_help_for_error(error)
print(help_text)
# Should provide AI-generated solution
```

### 4. Test GPU Masquerading (if you have GPU)
```python
import rtx50_compat
import torch

if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU capability: {cap}")
    # RTX 5090 should show (9, 0) instead of (12, 0)
```

### 5. Test Full Installation Flow
The package should help users through the complex xformers build:
```bash
# Set environment for RTX 5090
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0+PTX"
export RTX50_CHECK_INSTALL=1

python -c "import rtx50_compat"
# Should run SmartInstaller and provide guidance
```

## Questions for Gemini
1. Does the package install cleanly from PyPI?
2. Are the AI detection features working?
3. Does the SmartInstaller provide helpful guidance?
4. Any issues with the import hooks or error handling?
5. Suggestions for improving the AI assistance features?

## The Goal
Make RTX 5090 "just work" for everyone in the AI ecosystem. When users hit build errors with xformers/flash-attention, the package should automatically detect their AI assistant and generate fixes.

Please test this thoroughly and let me know if the AI-assisted installation works as intended!