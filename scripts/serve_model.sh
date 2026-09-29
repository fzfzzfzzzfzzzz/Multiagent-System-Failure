#!/usr/bin/env bash
# Run in tmux. Usage: bash serve_model.sh 8b 0,1 18151
# 14b: 2,3,4,5 / 18152; 27b on 158: 0,1,2,3 / 18158.
set -euo pipefail
MODEL=${1:?8b, 14b or 27b required}
GPUS=${2:?comma-separated idle GPU IDs required}
PORT=${3:?port required}
export CUDA_VISIBLE_DEVICES="$GPUS"
export TEAMLEARN_GPUS="$GPUS"
export PYTHONPATH=/data/fangc/teamlearn/gpu_tools${PYTHONPATH:+:$PYTHONPATH}
case "$MODEL" in
  8b) NAME=Qwen3-8B; PY=/data/fangc/envs/vllm-0.8.5/bin/python; ENGINE=vllm;;
  14b) NAME=Qwen3-14B; PY=/data/fangc/envs/vllm-0.8.5/bin/python; ENGINE=vllm;;
  27b) NAME=Qwen3.8-27B; PY=/data/fangc/envs/sglang-qwen38-260915/bin/python; ENGINE=sglang;;
  *) echo 'Unknown model' >&2; exit 2;;
esac
"$PY" -m gpustat --no-color
"$PY" - <<'PY'
import os,pynvml as n
n.nvmlInit()
for i in map(int,os.environ['TEAMLEARN_GPUS'].split(',')):
    if n.nvmlDeviceGetMemoryInfo(n.nvmlDeviceGetHandleByIndex(i)).used>512*1024**2:
        raise SystemExit(f'GPU {i} is occupied; refusing to start')
PY
TP=$(awk -F, '{print NF}' <<< "$GPUS")
if [ "$ENGINE" = vllm ]; then
  exec "$PY" -m vllm.entrypoints.openai.api_server --model "/data/fangc/models/$NAME" --served-model-name "$NAME" --host 127.0.0.1 --port "$PORT" --tensor-parallel-size "$TP" --gpu-memory-utilization 0.8 --max-model-len 32768 --max-num-seqs 4 --enforce-eager --disable-log-requests --generation-config vllm
else
  exec "$PY" -m sglang.launch_server --model-path "/data/fangc/models/$NAME" --served-model-name "$NAME" --host 127.0.0.1 --port "$PORT" --tp-size "$TP" --mem-fraction-static 0.8 --context-length 32768
fi
