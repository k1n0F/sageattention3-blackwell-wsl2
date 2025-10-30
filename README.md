'# SageAttention 3 — Blackwell (WSL2 Build Report)

**Author:** [k1n0f]  
**Based on:** [jt-zhang/SageAttention3](https://github.com/jt-zhang/SageAttention3)  
**Workflow author** [Super Comfy](https://www.youtube.com/channel/UCTg77WMzeyJGp1gV3TD30uA)
[Video: Fast & High Quality Wan 2.2 Image to Video Workflow!](https://www.youtube.com/watch?v=jn-4BR1UhEY)
[sonnybox/yt-files — COMFY/workflows/Wan 2.2 Image to Video.json](https://github.com/sonnybox/yt-files/blob/main/COMFY/workflows/Wan%202.2%20Image%20to%20Video.json)
**Environment:** WSL2 Ubuntu 22.04 • CUDA 13.0 • PyTorch 2.10.0-dev • RTX 5080 (Blackwell SM_120)

---

## Overview
This document outlines the complete process of compiling and stabilizing **SageAttention 3** under **WSL2 (Ubuntu 22.04)** with **CUDA 13.0**, **PyTorch 2.10.0-dev**, and an **NVIDIA RTX 5080 (Blackwell)**.  
The goal was to enable full functionality in **ComfyUI + WAN Video Wrapper**, achieve stable **FP4 kernel execution**, and eliminate allocator and CUDA graph errors.

---

## Proof of Success

```
attention mode override: sageattn3
tensor out (1, 8, 128, 64) torch.bfloat16 cuda:0
Max allocated memory: 9.953 GB
Comfy-VFI done — 125 frames generated at torch.Size([3, 1984, 1120])
Prompt executed in 341.08 seconds
2K VIDEO — 125 FRAMES — 5.7 MINUTES — UNDER WSL2
<img src="mosaic.png" alt="Mosaic Preview"> <img src="log.png" alt="Log">
```

---

## System Configuration

| Component | Version / Details |
|------------|------------------|
| GPU | NVIDIA GeForce RTX 5080 (Blackwell SM_120) |
| Host Driver | 560.35.03 (Linux) |
| WSL2 Kernel | 5.15.153.1-microsoft-standard-WSL2 |
| CUDA Toolkit | 13.0.52 |
| cuDNN | 9.5.1 |
| NCCL | 2.21.5 |
| PyTorch | 2.10.0.dev20251025 + cu130 (built from source) |
| TorchVision | 0.21.0a0 (nightly) |
| Python | 3.11.9 (venv) |
| GCC | 11.4.0 |
| CMake | 3.28.3 |
| OS | Ubuntu 22.04 LTS (WSL2) |
| ComfyUI | commit `6f14d91` (Oct 2025) |
| SageAttention3 Source | local branch from [jt-zhang/SageAttention3](https://github.com/jt-zhang/SageAttention3) |

---

## Objectives
1. Compile SageAttention3 for CUDA 13.0 / SM_120.  
2. Resolve all compilation, linking, and runtime errors.  
3. Validate FP4 quantization and Blackwell kernels.  
4. Achieve stable ComfyUI integration.

---

## Key Runtime Fixes

| Problem | Fix |
|----------|-----|
| `TypeError: NoneType + str` | `export CUDA_HOME=/usr/local/cuda-13.0` |
| `fatal error: Python.h` | `sudo apt install python3.10-dev` |
| `cannot find -lcuda` | `export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH` |
| `checkPoolLiveAllocations` | `PYTORCH_CUDA_ALLOC_CONF=backend:native,max_split_size_mb:128,expandable_segments:true` |
| `cudaMallocAsync != native` | `export PYTORCH_CUDA_ALLOC_CONF=backend:native` |
| `Unknown allocator backend` | Switched allocator backend to `native` |
| `Killed during decode` | Increased WSL2 RAM to 64 GB + 24 GB swap |
| `No module named sageattention` | Added minimal `__init__.py` shim files in `/sageattention3/`, `/sageattention3/blackwell/`, and `/sageattention3/csrc/` to expose the package during build and runtime import |

---

## Validation Summary

| Test | Result |
|------|---------|
| Module import | Successful |
| CUDA 13 detection | Verified |
| FP4 API | Detected |
| Tensor output | Valid |
| ComfyUI runtime | Stable |
| Allocator | Stable |
| CUDAGraph | Stable |
| Full 2K Video (125 frames) | **SUCCESS** |

---

## SageAttention3 API — Verified Functions

### FP4 Quantization
| Function | Description |
|-----------|-------------|
| `scale_and_quant_fp4(x)` | Converts BF16 → packed FP4 (uint8) + scale |
| `scale_and_quant_fp4_permute(x)` | Same as above with permuted layout for SM_120 |
| `scale_and_quant_fp4_transpose(x)` | Transposed FP4 quantization variant |
| `pythonscale_and_quant_fp4(x)` | Pure Python reference implementation |

### FP4 Attention Kernels
| Function | Description |
|-----------|-------------|
| `blockscaled_fp4_attn(Q, K, V, scale)` | Native Blackwell attention kernel (SM_120 Tensor Cores) |
| `fp4quant_cuda` / `fp4attn_cuda` | CUDA backend extensions for FP4 operations |

### Misc Utilities
| Function | Description |
|-----------|-------------|
| `dequant_fp4(q, s)` | Placeholder for FP4 → BF16 dequantization |
| `torch.compiler.cudagraph_mark_step_begin()` | Prevents CUDAGraph tensor overwrite between runs |

---

## Reproduction Guide

1. Start from a fresh WSL2 Ubuntu 22.04 instance.  
2. Install CUDA 13.0 and toolchain:  
   ```bash
   sudo apt install build-essential python3.11 python3.11-dev cmake git
   export CUDA_HOME=/usr/local/cuda-13.0
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   ```
3. Build PyTorch from source with CUDA 13.  
4. Clone SageAttention3 and CUTLASS.  
5. Add `__init__.py` (shim files) to all package roots.  
6. Fix linker:  
   ```bash
   sudo ln -s /usr/lib/wsl/lib/libcuda.so /usr/local/cuda/lib64/libcuda.so
   ```
7. Build and install:  
   ```bash
   TORCH_CUDA_ARCH_LIST="12.0+PTX" python setup.py bdist_wheel
   pip install dist/sageattention3_blackwell*.whl
   ```
8. Verify import:  
   ```bash
   python -c "import sageattention3; print('OK')"
   ```
9. Launch ComfyUI and ensure `ComfyUI-WanVideoWrapper` detects SageAttention3.  
10. **Run WAN Video Workflow** → Generates a **2K video in 5.7 minutes**.

---

## Conclusion
After extensive debugging and multiple rebuilds, **SageAttention 3** was successfully compiled for **NVIDIA Blackwell (SM_120)** with **CUDA 13.0** and **PyTorch 2.10-dev** under **WSL2**.  
All critical linking and runtime issues were resolved, FP4 kernel support verified, and the module runs stably within ComfyUI.

**FIRST FULL SAGEATTENTION3 INFERENCE ON BLACKWELL — UNDER WSL2, WITHOUT OFFICIAL SUPPORT.**

---

```markdown
**RTX 5080 + SageAttention3 — 2K VIDEO IN 5.7 MIN (WSL2, CUDA 13.0)**  
- 125 frames @ 1984×1120  
- VRAM usage: 9.95 GB  
- 6+ critical fixes  
- Repo: https://github.com/k1n0f/sageattention3-blackwell-wsl2  

```
