#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync,max_split_size_mb:192,expandable_segments:True"

source ~/ComfyUI/venv/bin/activate

export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

export XFORMERS_DISABLED=1
export PYTORCH_SDP_KERNEL=mem_efficient
export CUDA_LAUNCH_BLOCKING=0

env -u PYTORCH_CUDA_ALLOC_CONF \
python main.py \
  --bf16-unet --bf16-vae --bf16-text-enc \
  --disable-xformers \
  --reserve-vram 2560 \
  --lowvram \
  --listen 0.0.0.0 --port 8188
