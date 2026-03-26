# ============================================================
# Qwen3-ASR Service Docker 构建文件
# ============================================================
# 基础镜像：NVIDIA CUDA 12.8 + Ubuntu 22.04
# ============================================================
ARG CUDA_VERSION=12.8.0
ARG from=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04
FROM ${from} AS base

ARG DEBIAN_FRONTEND=noninteractive

# ============================================================
# 安装系统依赖
# ============================================================
RUN apt update -y && apt upgrade -y && apt install -y --no-install-recommends \
    git \
    git-lfs \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    ffmpeg \
    libsndfile1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# ============================================================
# 安装 Python 依赖
# ============================================================
# 官方推荐：使用 qwen-asr[vllm] 自动安装 vllm==0.14.0 + 全部依赖
RUN pip3 install -U pip setuptools wheel \
    && pip3 install -U "qwen-asr[vllm]"

# 可选：安装 flash-attn 加速推理（需要 GPU 支持）
# RUN pip3 install flash-attn

# ============================================================
# 复制应用代码
# ============================================================
COPY . .

# ============================================================
# 环境变量
# ============================================================
ENV QWEN_ASR_MODEL_PATH=/models/Qwen3-ASR-1.7B
ENV PORT=8002
ENV IDLE_TIMEOUT=600
ENV GPU_UTILIZATION=0.8

# ============================================================
# 端口
# ============================================================
EXPOSE 8002

# ============================================================
# 启动命令
# ============================================================
CMD ["python3", "app.py"]
