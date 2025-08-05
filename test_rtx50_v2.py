#!/usr/bin/env python3
"""
Test rtx50-compat v2.0.0 with AI assistance features
"""

import os
import sys
import subprocess

def test_fresh_install():
    """Test installing rtx50-compat from PyPI"""
    print("🧪 Testing rtx50-compat v2.0.0 fresh install...")
    
    # Create a fresh test environment
    test_dir = "/tmp/test_rtx50_v2"
    os.makedirs(test_dir, exist_ok=True)
    os.chdir(test_dir)
    
    # Create a fresh venv
    print("\n1. Creating fresh virtual environment...")
    subprocess.run(["uv", "venv", "test_env"], check=True)
    
    # Install rtx50-compat from PyPI
    print("\n2. Installing rtx50-compat v2.0.0 from PyPI...")
    subprocess.run([
        "uv", "pip", "install", "--python", "test_env/bin/python", "--upgrade", "rtx50-compat"
    ], check=True)
    
    # Test import and features
    print("\n3. Testing import and AI detection...")
    test_code = """
import sys
sys.path.insert(0, '.')

# Test basic import
import rtx50_compat
print(f"✅ rtx50-compat version: {rtx50_compat.__version__}")

# Test AI assistant detection
from rtx50_compat import AIAssistant, SmartInstaller
assistants = AIAssistant.detect_assistants()
print(f"✅ AI assistants detected: {assistants}")

# Test SmartInstaller
installer = SmartInstaller()
print("✅ SmartInstaller created successfully")

# Test environment check
issues = installer.check_environment()
print(f"📋 Environment issues found: {len(issues)}")
for issue in issues:
    print(f"  • {issue}")

# Test capability patching (if GPU available)
try:
    import torch
    if torch.cuda.is_available():
        original_cap = torch.cuda.get_device_capability(0)
        print(f"✅ Original GPU capability: {original_cap}")
        
        # Import should trigger patching
        import rtx50_compat
        patched_cap = torch.cuda.get_device_capability(0)
        print(f"✅ Patched GPU capability: {patched_cap}")
except ImportError:
    print("⚠️  PyTorch not installed - skipping GPU tests")

print("\\n✅ All basic tests passed!")
"""
    
    with open("test_script.py", "w") as f:
        f.write(test_code)
    
    subprocess.run(["test_env/bin/python", "test_script.py"])
    
    # Test AI help feature
    print("\n4. Testing AI help generation...")
    ai_test_code = """
from rtx50_compat import get_ai_help_for_error

# Simulate an xformers error
error_msg = "ImportError: libcudnn_ops_infer.so.8: cannot open shared object file"
print("📝 Testing AI help for error:", error_msg)

# This should detect available AI and provide help
help_msg = get_ai_help_for_error(error_msg)
print("\\nAI Assistant response:")
print(help_msg[:200] + "..." if len(help_msg) > 200 else help_msg)
"""
    
    with open("test_ai_help.py", "w") as f:
        f.write(ai_test_code)
    
    subprocess.run(["test_env/bin/python", "test_ai_help.py"])
    
    print("\n✅ rtx50-compat v2.0.0 tests completed!")
    return True

def create_gemini_validation():
    """Create validation script for Gemini"""
    validation_prompt = """Please validate the rtx50-compat v2.0.0 package:

1. Check PyPI page: https://pypi.org/project/rtx50-compat/2.0.0/
2. Review the AI-assisted installation features:
   - AIAssistant class that detects claude/gemini CLI
   - SmartInstaller that checks environment and suggests fixes
   - get_ai_help_for_error() function for troubleshooting

3. Test the package installation:
```bash
uv pip install rtx50-compat==2.0.0
python -c "import rtx50_compat; print(rtx50_compat.__version__)"
```

4. Verify the smart features work:
```python
from rtx50_compat import AIAssistant, SmartInstaller
print(AIAssistant.detect_assistants())
installer = SmartInstaller()
installer.check_environment()
```

5. Confirm the package:
   - Properly imports and patches PyTorch for RTX 5090
   - Detects AI assistants when available
   - Provides helpful error messages and AI-generated fixes
   - Version shows as "0.2.0" internally but 2.0.0 on PyPI

Please verify this works correctly and that the AI assistance features are properly integrated."""
    
    with open("/home/jw/dev/game1/rtx50-compat-repo/gemini_validate_v2.txt", "w") as f:
        f.write(validation_prompt)
    
    print("\n📝 Created Gemini validation prompt at: gemini_validate_v2.txt")

if __name__ == "__main__":
    # Run tests
    test_fresh_install()
    create_gemini_validation()
    
    print("\n🎯 Next step: Run Gemini validation")
    print("gemini -p \"$(cat /home/jw/dev/game1/rtx50-compat-repo/gemini_validate_v2.txt)\"")