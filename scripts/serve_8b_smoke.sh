#!/usr/bin/env bash
set -euo pipefail
mkdir -p /data/fangc/teamlearn/smoke
export CUDA_VISIBLE_DEVICES=6,7
export PYTHONPATH=/data/fangc/teamlearn/gpu_tools${PYTHONPATH:+:$PYTHONPATH}
PY=/data/fangc/envs/vllm-0.8.5/bin/python
$PY -m gpustat --no-color
$PY - <<'PY'
import pynvml as n
n.nvmlInit()
for i in [6,7]:
    h=n.nvmlDeviceGetHandleByIndex(i)
    if n.nvmlDeviceGetMemoryInfo(h).used > 512*1024**2:
        raise SystemExit(f'GPU {i} is not idle; refusing to start')
PY
exec "$PY" -m vllm.entrypoints.openai.api_server \
  --model /data/fangc/models/Qwen3-8B --served-model-name Qwen3-8B \
  --host 127.0.0.1 --port 18151 --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.60 --max-model-len 16384 \
  --max-num-seqs 4 --enforce-eager --disable-log-requests
