# 🚀 Install Qwen with vLLM on RTX 5090

## Quick Install with uv (2 minutes - FAST!)

```bash
# 1. Install uv if you haven't (one time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create a virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install everything with uv (MUCH faster than pip)
uv pip install rtx50-compat vllm transformers accelerate

# 4. Test it works
python3 -c "import rtx50_compat; import vllm; print('✅ Ready!')"
```

## Alternative: pip install (slower)

```bash
# Only if uv isn't available
pip install rtx50-compat
pip install vllm
pip install transformers accelerate
```

## Run Your First Qwen Model

```python
# Save this as test_qwen.py
import rtx50_compat  # Must be first!
from vllm import LLM, SamplingParams

# Load Qwen 7B (fits in your 32GB VRAM)
llm = LLM("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

# Generate
output = llm.generate(["Hello, tell me a joke"], SamplingParams(max_tokens=50))
print(output[0].outputs[0].text)
```

## Available Qwen Models for RTX 5090

### Fast Models (150-200 tokens/s)
- `Qwen/Qwen2.5-7B-Instruct` - General purpose (14GB VRAM)
- `Qwen/Qwen2.5-Coder-7B-Instruct` - Code specialist (14GB VRAM)

### Balanced Models (80-120 tokens/s)
- `Qwen/Qwen2.5-14B-Instruct` - Better quality (28GB VRAM)

### Maximum Quality (20-30 tokens/s)
- `Qwen/Qwen2.5-72B-Instruct-AWQ` - SOTA with 4-bit (30GB VRAM)
- `Qwen/Qwen2.5-72B-Instruct-GPTQ` - Alternative quant (30GB VRAM)

## Troubleshooting

### "No module named vllm"
```bash
# Always use uv - it's SO much faster
uv pip install vllm torch transformers
```

### "CUDA out of memory"
```python
# Reduce memory usage
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    gpu_memory_utilization=0.80,  # Use only 80% of VRAM
    max_model_len=4096,  # Reduce context length
)
```

### "No kernel image available"
```python
# Make sure rtx50_compat is imported FIRST
import rtx50_compat  # <-- This must be before everything else
import torch
from vllm import LLM
```

## Performance Tips

1. **Use 7B for speed**: 150-200 tokens/s for real-time apps
2. **Use 72B-AWQ for quality**: Still 20-30 tokens/s!
3. **Batch requests**: vLLM excels at batching
4. **Enable Flash Attention**: Already enabled by default

## Example: Code Generation with Qwen-Coder

```python
import rtx50_compat
from vllm import LLM, SamplingParams

# Load code-specific model
llm = LLM("Qwen/Qwen2.5-Coder-7B-Instruct", trust_remote_code=True)

prompt = """Write a Python FastAPI endpoint for user login with:
- JWT authentication
- Password hashing
- Input validation
- Error handling"""

output = llm.generate([prompt], SamplingParams(temperature=0.2, max_tokens=500))
print(output[0].outputs[0].text)
```

## What's Next?

1. Check out `rtx50-compat-repo/examples/qwen_vllm_setup.py` for more examples
2. Try different models based on your needs
3. Join the community and share your benchmarks!

---

**Remember**: You have 32GB VRAM. Use it! Even the 72B models work with quantization 🚀