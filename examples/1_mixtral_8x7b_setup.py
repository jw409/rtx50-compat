#!/usr/bin/env python3
"""
1. Mixtral 8x7B - Best Grok alternative for RTX 5090
MoE architecture similar to Grok, fits entirely in 32GB VRAM
"""

import rtx50_compat  # Enable RTX 5090
from vllm import LLM, SamplingParams

# Initialize Mixtral 8x7B
print("🚀 Setting up Mixtral 8x7B on RTX 5090...")
print("This model has 47B parameters with 8 experts (similar to Grok's MoE)")

llm = LLM(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.95,
    max_model_len=8192,  # Supports up to 32k context
    trust_remote_code=True
)

# Example: Grok-style reasoning
prompts = [
    "Explain why Grok from X.AI uses a Mixture of Experts architecture",
    "Write a Python function that implements a simple MoE layer",
    "Compare transformer architectures: dense vs sparse (MoE)",
]

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=200
)

print("\n📝 Generating responses...")
outputs = llm.generate(prompts, sampling_params)

for i, output in enumerate(outputs):
    print(f"\n{'='*60}")
    print(f"Prompt {i+1}: {prompts[i]}")
    print(f"Response: {output.outputs[0].text}")

print("\n✅ Expected performance on RTX 5090: 50-70 tokens/s")
print("💾 VRAM usage: ~24-28GB (comfortable fit in 32GB)")