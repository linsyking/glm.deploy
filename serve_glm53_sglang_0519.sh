#!/usr/bin/env bash
# Validated SGLang 0.5.19 production launch for GLM-5.3 on both GB200 nodes.
set -euo pipefail
source /scratch_local/user_data/yiming/serving/output_policy.env
export SGLANG_HIDE_THINKING SGLANG_TEXT_ONLY SGLANG_WEB_SEARCH TAVILY_API_KEY
export PYTHONPATH="/scratch_local/user_data/yiming/serving${PYTHONPATH:+:$PYTHONPATH}"

ROLE=${1:-head}
PORT=${2:-8000}
case "$ROLE" in head|worker) ;; *) echo "Usage: $0 head|worker [port]" >&2; exit 2 ;; esac

readonly TEST_ENV=/scratch_local/user_data/yiming/sglang_env_0519
readonly MODEL_PATH=/scratch_local/user_data/yiming/models/GLM-5.3
readonly HEAD_IP=10.78.202.30
readonly DIST_PORT=5000
readonly DIST_INIT_ADDR="${HEAD_IP}:${DIST_PORT}"
readonly NNODES=2

export TMPDIR=/scratch_local/user_data/yiming/tmp
export HF_HOME=/scratch_local/user_data/yiming/.cache/huggingface
export XDG_CACHE_HOME=/scratch_local/user_data/yiming/.cache
export SGLANG_CACHE_DIR=/scratch_local/user_data/yiming/.cache/sglang
export SGLANG_JIT_CACHE_DIR="$SGLANG_CACHE_DIR/jit"
export SGLANG_DG_CACHE_DIR="$SGLANG_CACHE_DIR/deep_gemm"
# DeepGEMM reads this variable when its package is first imported, before
# SGLang's wrapper gets a chance to mirror SGLANG_DG_CACHE_DIR into it.
export DG_JIT_CACHE_DIR="$SGLANG_DG_CACHE_DIR"
mkdir -p "$TMPDIR" "$HF_HOME" "$XDG_CACHE_HOME" \
    "$SGLANG_CACHE_DIR" "$SGLANG_JIT_CACHE_DIR" "$SGLANG_DG_CACHE_DIR"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

export CUDA_HOME=/scratch_local/user_data/yiming/cuda-13.3
export PATH="$CUDA_HOME/bin:$TEST_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$TEST_ENV/lib/python3.12/site-packages/nvidia/nccl/lib:$TEST_ENV/lib/python3.12/site-packages/torch/lib:$TEST_ENV/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:$TEST_ENV/lib/python3.12/site-packages/nvidia/cublas/lib:$TEST_ENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$TEST_ENV/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$TEST_ENV/lib/python3.12/site-packages/nvidia/nccl/lib:$CUDA_HOME/lib64:${LIBRARY_PATH:-}"

export SGLANG_NCCL_SO_PATH="$TEST_ENV/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2"
export LD_PRELOAD="$SGLANG_NCCL_SO_PATH"
export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0
export TP_SOCKET_IFNAME=bond0
export NCCL_DEBUG=WARN
unset NCCL_DEBUG_SUBSYS
export NCCL_NET_GDR_LEVEL=5
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5
export NCCL_CROSS_NIC=1
export NCCL_CUMEM_ENABLE=1
export NCCL_NVLS_ENABLE=1
export CC=/usr/bin/gcc-13
export CXX=/usr/bin/g++-13
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=False

if [[ "$ROLE" == head ]]; then
    NODE_RANK=0
else
    NODE_RANK=1
fi

exec "$TEST_ENV/bin/python3" -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --served-model-name glm-5.3 \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --tp 8 \
    --dist-init-addr "$DIST_INIT_ADDR" \
    --nnodes "$NNODES" \
    --node-rank "$NODE_RANK" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --kv-cache-dtype fp8_e4m3 \
    --speculative-algorithm NEXTN \
    --moe-runner-backend cutlass \
    --speculative-moe-runner-backend cutlass \
    --enable-symm-mem \
    --enable-nccl-nvls \
    --disable-flashinfer-autotune \
    --mem-fraction-static 0.85 \
    --enable-metrics \
    --enable-cache-report
