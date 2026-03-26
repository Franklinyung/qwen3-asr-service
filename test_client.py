import requests
import os
import sys
import time

# 默认配置
BASE_URL = os.getenv("ASR_BASE_URL", "http://localhost:8002")
# TEST_AUDIO = "/home/ykl/.openclaw/workspace-code-dev/shenlun-assistant/data/audio/3ddc9f6c.mp3"
# TEST_AUDIO = "/tmp/test_asr_bvid.mp3"
TEST_AUDIO = "/tmp/BV1k9PHz8ERM.m4a"# 超长音频

def test_health():
    """测试健康检查接口"""
    print("\n[1] 测试健康检查接口...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print(f"✅ 健康检查通过: {response.json()}")
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 健康检查请求异常: {e}")

def test_transcription(audio_path):
    """测试语音转文字接口"""
    print(f"\n[2] 测试语音转文字接口 (文件: {os.path.basename(audio_path)})...")
    
    if not os.path.exists(audio_path):
        print(f"❌ 音频文件不存在: {audio_path}")
        return

    files = {
        'file': (os.path.basename(audio_path), open(audio_path, 'rb'), 'audio/mpeg')
    }
    data = {
        'language': 'Chinese'
    }

    start_time = time.time()
    try:
        print("   (正在转录，首次调用可能需要加载模型，请耐心等待...)")
        response = requests.post(
            f"{BASE_URL}/v1/audio/transcriptions", 
            files=files, 
            data=data
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 转录成功! (耗时: {elapsed:.2f}s)")
            print(f"   语言: {result['language']}")
            print(f"   文字 (前100字): {result['text'][:100]}...")
            print(f"   推理时长: {result['duration']}s")
            print(f"   使用模型: {result['model']}")
        else:
            print(f"❌ 转录失败: {response.status_code}")
            print(f"   详情: {response.text}")
    except Exception as e:
        print(f"❌ 转录请求异常: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("Qwen3-ASR FastAPI 服务客户端测试")
    print("=" * 60)
    
    # 检查服务是否运行
    test_health()
    
    # 运行转录测试
    test_transcription(TEST_AUDIO)
    
    print("\n" + "=" * 60)
    print("测试结束。")
