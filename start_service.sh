#!/bin/bash
# Qwen3-ASR Service 启动脚本

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 激活 conda 环境（根据实际情况修改环境名称）
# conda activate asr_test
# 或者用 conda run：
# conda run -n asr_test python app.py

# 直接运行（推荐先创建 conda 环境）
pip install -r requirements.txt
python app.py
