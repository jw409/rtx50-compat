#!/usr/bin/env python3
"""
Run Qwen models locally with vLLM on RTX 5090
Qwen models are excellent for code generation and multilingual tasks
"""

import rtx50_compat  # Enable RTX 5090 support
import torch
from vllm import LLM, SamplingParams

def setup_qwen_models():
    """Setup various Qwen models with vLLM"""
    
    print("🚀 Qwen Model Setup for RTX 5090 with vLLM")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()
    
    # Qwen model options for RTX 5090 (32GB VRAM)
    qwen_models = {
        "Qwen/Qwen2.5-7B-Instruct": {
            "size": "7B",
            "vram_needed": "14-16 GB",
            "expected_speed": "150-200 tokens/s",
            "best_for": "Fast inference, coding, general chat"
        },
        "Qwen/Qwen2.5-14B-Instruct": {
            "size": "14B", 
            "vram_needed": "28-30 GB",
            "expected_speed": "80-120 tokens/s",
            "best_for": "Balance of speed and quality"
        },
        "Qwen/Qwen2.5-32B-Instruct": {
            "size": "32B",
            "vram_needed": "32GB+ (needs quantization)",
            "expected_speed": "40-60 tokens/s",
            "best_for": "High quality, complex reasoning"
        },
        "Qwen/Qwen2.5-72B-Instruct-AWQ": {
            "size": "72B AWQ",
            "vram_needed": "30-32 GB",
            "expected_speed": "20-30 tokens/s", 
            "best_for": "Maximum quality with quantization"
        },
        "Qwen/Qwen2.5-Coder-7B-Instruct": {
            "size": "7B Coder",
            "vram_needed": "14-16 GB",
            "expected_speed": "150-200 tokens/s",
            "best_for": "Specialized for code generation"
        }
    }
    
    print("📋 Available Qwen Models for RTX 5090:")
    for model_id, info in qwen_models.items():
        print(f"\n{model_id}")
        print(f"  • Size: {info['size']}")
        print(f"  • VRAM: {info['vram_needed']}")
        print(f"  • Speed: {info['expected_speed']}")
        print(f"  • Best for: {info['best_for']}")

def run_qwen_7b():
    """Run Qwen2.5-7B - Fastest option, fits comfortably in VRAM"""
    
    print("\n" + "="*60)
    print("🏃 Running Qwen2.5-7B-Instruct")
    print("="*60)
    
    # Initialize model
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,  # Conservative for stability
        trust_remote_code=True,  # Required for Qwen models
        max_model_len=8192,
    )
    
    # Test prompts
    prompts = [
        "Write a Python function to implement binary search with detailed comments",
        "Explain quantum computing in simple terms",
        "你好！请用中文解释什么是机器学习",  # Multilingual test
    ]
    
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=200,
    )
    
    print("\n📝 Generating responses...")
    outputs = llm.generate(prompts, sampling_params)
    
    for i, output in enumerate(outputs):
        print(f"\n{'='*60}")
        print(f"Prompt: {prompts[i]}")
        print(f"Response: {output.outputs[0].text}")
    
    print("\n✅ Expected performance: 150-200 tokens/s")
    print("💾 VRAM usage: ~14-16 GB")

def run_qwen_72b_awq():
    """Run Qwen2.5-72B with AWQ quantization - Maximum quality"""
    
    print("\n" + "="*60)
    print("🧠 Running Qwen2.5-72B-Instruct-AWQ (Quantized)")
    print("="*60)
    
    # Initialize large model with quantization
    llm = LLM(
        model="Qwen/Qwen2.5-72B-Instruct-AWQ",
        quantization="awq",  # 4-bit quantization
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,  # Use most VRAM
        trust_remote_code=True,
        max_model_len=4096,  # Reduce context for memory
    )
    
    # Complex reasoning prompt
    prompt = """Analyze the following code and suggest improvements:

def calculate_fibonacci(n):
    if n <= 1:
        return n
    else:
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

Consider: performance, readability, and best practices."""
    
    sampling_params = SamplingParams(
        temperature=0.3,  # Lower temp for code analysis
        top_p=0.95,
        max_tokens=300,
    )
    
    print("\n📝 Generating expert analysis...")
    output = llm.generate([prompt], sampling_params)[0]
    
    print(f"\nPrompt: {prompt}")
    print(f"\nResponse: {output.outputs[0].text}")
    
    print("\n✅ Expected performance: 20-30 tokens/s")
    print("💾 VRAM usage: ~30-32 GB (near limit)")

def run_qwen_coder():
    """Run Qwen2.5-Coder - Specialized for code generation"""
    
    print("\n" + "="*60)
    print("💻 Running Qwen2.5-Coder-7B-Instruct")
    print("="*60)
    
    llm = LLM(
        model="Qwen/Qwen2.5-Coder-7B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        max_model_len=16384,  # Longer context for code
    )
    
    # Code generation prompt
    prompt = """Create a FastAPI application with:
1. User authentication (JWT)
2. CRUD operations for a Task model
3. PostgreSQL database
4. Proper error handling
Include all necessary imports and setup."""
    
    sampling_params = SamplingParams(
        temperature=0.2,  # Low temp for code
        top_p=0.95,
        max_tokens=500,
    )
    
    print("\n📝 Generating code...")
    output = llm.generate([prompt], sampling_params)[0]
    
    print(f"\nPrompt: {prompt}")
    print(f"\nGenerated Code:\n{output.outputs[0].text}")
    
    print("\n✅ Expected performance: 150-200 tokens/s")
    print("💾 VRAM usage: ~14-16 GB")

if __name__ == "__main__":
    # First, ensure rtx50_compat is working
    import rtx50_compat
    
    print("🔧 RTX 5090 Compatibility Check")
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        capability = torch.cuda.get_device_capability(0)
        print(f"✅ Capability: {capability} (masqueraded)")
    
    # Show available models
    setup_qwen_models()
    
    # Choose which model to run
    print("\n" + "="*60)
    print("Choose a model to run:")
    print("1. Qwen2.5-7B (Fast, fits easily)")
    print("2. Qwen2.5-72B-AWQ (Maximum quality)")
    print("3. Qwen2.5-Coder-7B (Code specialist)")
    print("\nNote: This is a demo. In practice, uncomment the model you want.")
    
    # Uncomment the model you want to run:
    # run_qwen_7b()         # Fast, general purpose
    # run_qwen_72b_awq()    # Maximum quality
    # run_qwen_coder()      # Code generation
    
    print("\n💡 Installation:")
    print("pip install rtx50-compat vllm transformers")
    print("\n🚀 Then uncomment the model you want in this script!")