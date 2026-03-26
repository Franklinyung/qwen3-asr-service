#!/bin/bash
# Qwen3-ASR Service 启动脚本

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 方式一：使用 conda 环境
# conda create -n asr_test python=3.10 -y
# conda activate asr_test
# pip install -r requirements.txt
# python app.py

# 方式二：直接运行（确保 Python 3.10+）
pip install -r requirements.txt
python app.py
