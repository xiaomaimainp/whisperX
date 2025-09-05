import torch
import warnings
import sys
import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# 禁用警告
warnings.filterwarnings("ignore")

def transcribe_audio_file(audio_path):
    """转录真实音频文件"""
    
    if not os.path.exists(audio_path):
        print(f"❌ 错误：找不到音频文件 {audio_path}")
        return None
    
    print("🎤 WhisperX 实际音频转录")
    print("=" * 60)
    print(f"📁 音频文件: {audio_path}")
    
    # 设备配置
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3-turbo"
    
    print(f"📱 设备: {device}")
    print(f"🔧 模型: {model_id}")
    
    # 加载模型
    print("⏳ 正在加载模型...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True
    )
    model.to(device)
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    # 创建管道
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    print("🎵 开始转录...")
    
    # 转录音频文件
    result = pipe(
        audio_path,
        generate_kwargs={"task": "transcribe"},  # 使用 transcribe 而不是 translate
        return_timestamps=True
    )
    
    print("\n" + "=" * 60)
    print("📝 转录结果:")
    print("=" * 60)
    print(result["text"])
    
    # 如果有时间戳信息，显示分段结果
    if "chunks" in result and result["chunks"]:
        print("\n" + "=" * 60)
        print("⏰ 时间戳分段:")
        print("=" * 60)
        for i, chunk in enumerate(result["chunks"], 1):
            start_time = chunk['timestamp'][0] if chunk['timestamp'][0] is not None else 0
            end_time = chunk['timestamp'][1] if chunk['timestamp'][1] is not None else "未知"
            print(f"[{i:2d}] {start_time:6.2f}s - {end_time:>6}s: {chunk['text'].strip()}")
    
    print("\n✅ 转录完成!")
    return result["text"]

if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) != 2:
        print("使用方法: python transcribe_real_audio.py <音频文件路径>")
        print("支持的格式: .wav, .mp3, .flac, .m4a, .ogg 等")
        print("示例: python transcribe_real_audio.py uploads/my_audio.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    transcribe_audio_file(audio_file)