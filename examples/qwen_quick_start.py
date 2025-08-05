#!/usr/bin/env python3
"""
Quick start: Run Qwen 7B with vLLM on RTX 5090
This is the simplest way to get started
"""

# Step 1: Enable RTX 5090 support
import rtx50_compat

# Step 2: Import vLLM
from vllm import LLM, SamplingParams

# Step 3: Load Qwen model (7B fits easily in 32GB)
print("Loading Qwen2.5-7B-Instruct...")
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    gpu_memory_utilization=0.90,
    trust_remote_code=True,
)

# Step 4: Generate text
prompt = "Write a Python function to calculate factorial"
sampling_params = SamplingParams(temperature=0.7, max_tokens=200)

print(f"\nPrompt: {prompt}")
print("\nGenerating...")

output = llm.generate([prompt], sampling_params)[0]
print(f"\nResponse:\n{output.outputs[0].text}")

print("\n✅ Success! Qwen is running locally on your RTX 5090!")
print("💡 Try other prompts or check qwen_vllm_setup.py for more models")