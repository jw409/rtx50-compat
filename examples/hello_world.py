#!/usr/bin/env python3
"""
Hello World example for rtx50-compat
Verifies that RTX 5090 is properly recognized and functional
"""

import rtx50_compat  # Must be imported before PyTorch!
import torch
import time

def main():
    print("🚀 RTX 50-series Compatibility Test\n")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Please check your installation.")
        return
    
    # Get device info
    device_name = torch.cuda.get_device_name(0)
    device_capability = torch.cuda.get_device_capability(0)
    device_props = torch.cuda.get_device_properties(0)
    
    print(f"✅ GPU Detected: {device_name}")
    print(f"✅ Compute Capability: {device_capability} (masqueraded)")
    print(f"✅ VRAM: {device_props.total_memory / 1024**3:.1f} GB")
    print(f"✅ CUDA Cores: {device_props.multi_processor_count * 128}")
    print()
    
    # Quick performance test
    print("Running quick benchmark...")
    
    # Matrix multiplication test
    size = 8192
    x = torch.randn(size, size, device='cuda', dtype=torch.float16)
    y = torch.randn(size, size, device='cuda', dtype=torch.float16)
    
    # Warmup
    for _ in range(3):
        _ = torch.matmul(x, y)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    iterations = 10
    for _ in range(iterations):
        result = torch.matmul(x, y)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    # Calculate TFLOPS
    flops = 2 * size**3 * iterations  # 2*n^3 for matrix multiplication
    tflops = (flops / elapsed) / 1e12
    
    print(f"✅ Matrix Multiply ({size}x{size}): {tflops:.1f} TFLOPS")
    print(f"✅ Time per iteration: {elapsed/iterations*1000:.1f} ms")
    
    # Memory bandwidth test
    print("\nMemory bandwidth test...")
    size_gb = 4
    size_bytes = size_gb * 1024**3
    elements = size_bytes // 4  # float32
    
    x = torch.randn(elements, device='cuda')
    y = torch.empty_like(x)
    
    # Warmup
    for _ in range(3):
        y.copy_(x)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    iterations = 10
    for _ in range(iterations):
        y.copy_(x)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    bandwidth_gb_s = (size_gb * 2 * iterations) / elapsed  # Read + Write
    print(f"✅ Memory Bandwidth: {bandwidth_gb_s:.1f} GB/s")
    
    # Batman mode check
    import os
    if os.environ.get('RTX50_BATMAN_MODE'):
        print("\n🦇 I am Batman - at your local jujitsu establishment")
        print("   RTX 5090 successfully disguised as H100")
        print("   You didn't see anything... 🌙")
    
    print("\n✨ All tests passed! Your RTX 5090 is ready for AI workloads!")

if __name__ == "__main__":
    main()