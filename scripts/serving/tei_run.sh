#!/bin/bash
# Launch a TEI (text-embeddings-inference) server for the sentence embedder,
# exposing an OpenAI-compatible endpoint at http://localhost:${PORT}/v1.
#
# The training pipeline calls it via the openai client when
# FinetuneConfig.sentence_embedder_backend == "tei" (RemoteEmbedder). This moves
# the MiniLM transformer forward out of the data workers into one batched GPU
# service, so the workers become thin HTTP clients.
#
# Run this in its own terminal (like scripts/serving/vllm_run.sh). GPU is the free RTX
# 4090 = nvidia-smi index 2 (Ada / sm_89 → the "89" image tag).
#
# After it prints "Ready", validate with:
#   /home/amax/.conda/envs/rpt/bin/python scripts/serving/tei_smoke.py
set -euo pipefail

MODEL="${MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
PORT="${PORT:-8080}"
GPU="${GPU:-2}"                       # nvidia-smi GPU index (free 4090)
# Pick the TEI image by the target GPU's arch unless IMAGE is set explicitly
# (sm_89 kernels won't run on an sm_80 A800, so the tag must match the card).
_gpu_name="$(nvidia-smi --id="${GPU}" --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
case "${_gpu_name}" in
    *4090*|*4080*|*L40*|*Ada*)  _tag="89-1.7" ;;      # Ada (sm_89)
    *A100*|*A800*|*A30*)        _tag="1.7" ;;         # Ampere sm_80
    *A10*|*A40*|*3090*|*A6000*) _tag="86-1.7" ;;      # Ampere sm_86
    *H100*|*H800*)              _tag="hopper-1.7" ;;
    *)                          _tag="1.7" ;;         # safe default
esac
IMAGE="${IMAGE:-ghcr.io/huggingface/text-embeddings-inference:${_tag}}"
echo "[tei] GPU ${GPU} = ${_gpu_name:-unknown} -> image ${IMAGE}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

# Verify the target GPU first (avoid landing on a busy/full card):
nvidia-smi --id="${GPU}" --query-gpu=name,memory.used,memory.total --format=csv,noheader || true

docker run --rm -it \
    --gpus "\"device=${GPU}\"" \
    -p "${PORT}:80" \
    -v "${HF_CACHE}/hub:/data" \
    -e HF_HUB_OFFLINE=1 \
    "${IMAGE}" \
    --model-id "${MODEL}" \
    --pooling mean \
    --auto-truncate \
    --max-client-batch-size 256 \
    --max-batch-tokens 65536

# CPU fallback (no GPU): set IMAGE=ghcr.io/huggingface/text-embeddings-inference:cpu-1.7
# and drop the --gpus flag.
