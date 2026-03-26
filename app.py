import os
import time
import torch
import gc
import uuid
import shutil
import asyncio
import logging
import subprocess
import glob
from logging.handlers import RotatingFileHandler
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# --- 日志配置 ---
LOG_DIR = os.getenv("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# 配置根日志记录器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(), # 控制台输出
        RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8') # 文件记录，10MB 轮转，保留5个
    ]
)
logger = logging.getLogger("qwen3-asr-service")

app = FastAPI(
    title="Qwen3-ASR vLLM Service", 
    description="OpenAI-compatible ASR API with auto GPU memory cleanup and long audio chunking",
    version="1.1.0"
)

# --- 配置参数 ---
MODEL_PATH = os.getenv("QWEN_ASR_MODEL_PATH", "/nas_share/users/ykl/downloads/models/Qwen3-ASR-1.7B")
GPU_UTILIZATION = float(os.getenv("GPU_UTILIZATION", "0.8"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "32768"))
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", "600")) # 10分钟空闲自动释放
TEMP_DIR = os.getenv("TEMP_DIR", "temp_audio")

# 分段配置
CHUNK_THRESHOLD = int(os.getenv("CHUNK_THRESHOLD", "600")) # 超过10分钟则分段
CHUNK_DURATION = int(os.getenv("CHUNK_DURATION", "300"))   # 每5分钟一段

os.makedirs(TEMP_DIR, exist_ok=True)

# --- 工具函数 ---
def get_audio_duration(file_path):
    """获取音频时长 (秒)"""
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', file_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return float(result.stdout.strip())
    except Exception as e:
        logger.error(f"获取音频时长失败: {e}")
        return 0

def split_audio(file_path, output_dir, segment_time=300):
    """将音频标准化并分段 (16kHz, Mono, 压缩质量以加快速度)"""
    try:
        # file_id 用于生成唯一文件名
        file_id = os.path.basename(file_path).split('.')[0]
        output_pattern = os.path.join(output_dir, f"{file_id}_chunk_%03d.wav")
        
        # 标准化参数：
        # -ar 16000: 采样率 16kHz (ASR 模型标准)
        # -ac 1: 单声道 (ASR 模型标准)
        # -f segment: 分段输出
        # -segment_time: 每段时长
        # 使用 wav 格式可以避免 mp3 编码带来的额外开销和切点不准
        cmd = [
            'ffmpeg', '-y', '-i', file_path, 
            '-ar', '16000', 
            '-ac', '1',
            '-f', 'segment', 
            '-segment_time', str(segment_time), 
            output_pattern
        ]
        logger.info(f"正在转换并分段音频: {file_path}")
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 返回生成的文件列表
        chunks = sorted(glob.glob(os.path.join(output_dir, f"{file_id}_chunk_*.wav")))
        return chunks
    except Exception as e:
        logger.error(f"音频转换或分段失败: {e}")
        return []

# --- 模型管理器 (带自动回收逻辑) ---
class ModelManager:
    def __init__(self):
        self.model = None
        self.last_used = 0
        self.lock = asyncio.Lock()

    async def get_model(self):
        async with self.lock:
            self.last_used = time.time()
            if self.model is None:
                # 延迟导入，防止启动时卡住
                logger.info("正在导入 qwen_asr 模块...")
                try:
                    from qwen_asr import Qwen3ASRModel
                except ImportError as e:
                    logger.error(f"导入 qwen_asr 失败: {e}")
                    raise HTTPException(status_code=500, detail="qwen-asr package not installed")

                logger.info(f"正在加载模型: {MODEL_PATH} ...")
                try:
                    # 针对 1.7B 优化的加载参数，参考 test_qwen_asr.py
                    self.model = Qwen3ASRModel.LLM(
                        model=MODEL_PATH,
                        gpu_memory_utilization=GPU_UTILIZATION,
                        max_model_len=MAX_MODEL_LEN,
                    )
                    logger.info("模型加载完成。")
                except Exception as e:
                    logger.error(f"模型加载失败: {str(e)}")
                    raise e
            return self.model

    async def check_idle(self):
        """后台任务：检查是否长时间未用并释放显存"""
        while True:
            await asyncio.sleep(60) # 每分钟检查一次
            if self.model and (time.time() - self.last_used > IDLE_TIMEOUT):
                async with self.lock:
                    if self.model and (time.time() - self.last_used > IDLE_TIMEOUT):
                        logger.info(f"检测到模型已空闲 {IDLE_TIMEOUT}s，正在释放显存...")
                        try:
                            # 释放资源前尝试销毁分布式进程组 (参考 test_qwen_asr.py)
                            if torch.distributed.is_initialized():
                                torch.distributed.destroy_process_group()
                            
                            self.model = None
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            logger.info("显存及分布式资源已回收。")
                        except Exception as e:
                            logger.error(f"资源回收过程中出错: {e}")
                            self.model = None # 仍然置空以防下次加载

manager = ModelManager()

# --- API 定义 ---
class TranscriptionResponse(BaseModel):
    text: str
    language: str
    duration: Optional[float] = None
    model: str = "Qwen3-ASR-1.7B"
    is_chunked: bool = False

@app.on_event("startup")
async def startup_event():
    # 启动后台空闲检查任务
    asyncio.create_task(manager.check_idle())
    
    # 启动时清理残留的临时文件
    if os.path.exists(TEMP_DIR):
        logger.info(f"正在清理临时目录: {TEMP_DIR} ...")
        for f in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, f)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.error(f"清理路径 {file_path} 失败: {e}")
                
    logger.info("服务已启动，空闲回收机制已激活。访问 /health 查看状态。")

@app.get("/")
async def root():
    return {"message": "Qwen3-ASR vLLM Service is running. Visit /docs for API documentation."}

@app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def transcribe(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: str = "Chinese"
):
    """
    语音转文字接口 (支持超长音频分段处理)
    """
    # 1. 保存临时音频文件
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1] or ".mp3"
    temp_file = os.path.join(TEMP_DIR, f"{file_id}{ext}")
    
    logger.info(f"接收到文件: {file.filename} -> {temp_file}")
    
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 辅助清理函数
    def cleanup_all(paths: List[str]):
        for p in paths:
            try:
                if os.path.exists(p):
                    if os.path.isfile(p): os.remove(p)
                    else: shutil.rmtree(p)
            except: pass

    try:
        # 2. 获取时长并决定是否分段
        duration = get_audio_duration(temp_file)
        logger.info(f"音频时长: {duration:.2f}s")
        
        # 获取模型
        model = await manager.get_model()
        
        final_text = ""
        is_chunked = False
        start_inference = time.time()

        if duration > CHUNK_THRESHOLD:
            is_chunked = True
            logger.info(f"音频超过阈值 {CHUNK_THRESHOLD}s，启动分段处理...")
            
            # 创建分段专用目录
            chunk_dir = os.path.join(TEMP_DIR, f"chunks_{file_id}")
            os.makedirs(chunk_dir, exist_ok=True)
            
            # 分段
            chunks = split_audio(temp_file, chunk_dir, CHUNK_DURATION)
            if not chunks:
                raise HTTPException(status_code=500, detail="Audio splitting failed")
            
            logger.info(f"成功切分为 {len(chunks)} 个片段")
            
            # 逐个处理
            texts = []
            for i, chunk_path in enumerate(chunks):
                logger.info(f"正在处理片段 [{i+1}/{len(chunks)}]: {os.path.basename(chunk_path)}")
                chunk_results = model.transcribe(audio=chunk_path, language=language)
                if chunk_results:
                    texts.append(chunk_results[0].text)
            
            final_text = "".join(texts)
            # 任务结束后清理分段目录
            background_tasks.add_task(cleanup_all, [temp_file, chunk_dir])
        else:
            # 普通处理
            logger.info(f"正在处理短音频...")
            results = model.transcribe(audio=temp_file, language=language)
            if not results:
                raise HTTPException(status_code=500, detail="Transcription failed")
            final_text = results[0].text
            background_tasks.add_task(cleanup_all, [temp_file])

        inference_time = time.time() - start_inference
        logger.info(f"转录完成，总耗时: {inference_time:.2f}s")
        
        return TranscriptionResponse(
            text=final_text,
            language=language,
            duration=round(inference_time, 2),
            model=os.path.basename(MODEL_PATH),
            is_chunked=is_chunked
        )

    except Exception as e:
        cleanup_all([temp_file])
        logger.error(f"转录过程发生异常: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """健康检查接口，返回模型加载状态"""
    return {
        "status": "healthy",
        "model_loaded": manager.model is not None,
        "gpu_utilization": GPU_UTILIZATION,
        "idle_timeout": IDLE_TIMEOUT,
        "last_used_at": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(manager.last_used)) if manager.last_used > 0 else "never"
    }

if __name__ == "__main__":
    import uvicorn
    # 获取端口配置
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
