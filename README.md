# Qwen3-ASR vLLM Service

Self-hosted, high-performance ASR inference service powered by **FastAPI** and **vLLM**.  
基于 FastAPI 和 vLLM 构建的高性能本地语音转文字（ASR）微服务。

---

## ✨ Features / 特性

- **🎯 OpenAI Compatible**: Provides `/v1/audio/transcriptions` endpoint. / 提供兼容 OpenAI 格式的接口。
- **⚡ High Performance**: Leveraging vLLM for lightning-fast inference on NVIDIA GPUs. / 使用 vLLM 实现极速 GPU 推理。
- **🧠 Smart Memory Management**: 
    - Auto GPU memory cleanup after idle timeout. / 支持空闲自动释放显存。
    - Optimized for Qwen3-ASR-1.7B to prevent OOM. / 针对 1.7B 模型优化，防止显存溢出。
- **🎞️ Long Audio Support**: 
    - Intelligent chunking for audios over 10 minutes (prevents memory spikes). / 智能分段处理超长音频（防止内存爆炸）。
    - Standardizes audio via FFmpeg (supports `.m4a`, `.wav`, `.mp3`, etc.). / 自动音频标准化，支持多种格式。
- **📝 Production Ready**: Includes shell startup scripts, rotating logs, and systemd templates. / 包含启动脚本、轮转日志及 Systemd 模板。

---

## 🛠️ Setup / 快速启动

### 1. Environment / 环境准备
```bash
conda create -n asr_test python=3.10 -y
conda activate asr_test

# 安装全部依赖（推荐，自动安装 qwen-asr + vllm==0.14.0）
pip install -r requirements.txt

# 或手动安装（使用官方 qwen-asr 包）
pip install "qwen-asr[vllm]"  # 官方推荐，自动处理版本兼容
pip install fastapi uvicorn[standard] python-multipart pydantic python-dotenv httpx
```

### 2. Configuration / 配置
```bash
cp .env.example .env
# Edit .env to set your model path / 修改 .env 中的模型路径
```

### 3. Run / 运行
```bash
# Using startup script (Recommended) / 使用启动脚本（推荐）
bash start_service.sh

# Or directly / 或直接运行
python app.py
```

---

## 📖 API Documentation / 接口文档

Refer to [API_DOCS.md](./API_DOCS.md) for detailed technical specifications.  
详细技术细节请参考 [API_DOCS.md](./API_DOCS.md)。

---

## 🧹 Maintenance / 自动维护说明

- **File Cleanup**: Temporary audio files are deleted immediately after transcription. / 转录完成后立即物理删除音频文件。
- **GPU Recycling**: The model is uninstalled from VRAM after 10 minutes of inactivity. / 10 分钟无请求自动释放显存。

---

## 📄 License
Open-sourced under the MIT License.
