from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import librosa
from opencc import OpenCC

app=FastAPI()


# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 指定保存文件的目录
UPLOAD_DIR="static/audioFiles"

# 确保上传目录存在
os.makedirs(UPLOAD_DIR, exist_ok=True)

## 支持的音频 MIME 类型
SUPPORTED_AUDIO_MIME_TYPES = [
    "audio/wav",
    "audio/x-wav",
    "audio/webm",
    "audio/mpeg",
    "audio/mp3",
    "audio/aac",
    "audio/ogg",
    "audio/flac",
]


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

#创建管道
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

#获取音频文件的持续时间（秒）
def get_audio_duration(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return librosa.get_duration(y=audio, sr=sr)


@app.post("/whisper/upload")
async def upload_audio(file: UploadFile = File(...)):
    try:
        # 检查文件类型
        content_type = file.content_type
        if content_type not in SUPPORTED_AUDIO_MIME_TYPES:
            raise HTTPException(status_code=400, detail="Unsupported file type. Only audio files are allowed.")

        # 生成文件保存路径
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        # 保存上传的文件
        with open(file_location, "wb+") as file_object:
            file_object.write(await file.read())

         # 获取音频文件的持续时间
        duration = get_audio_duration(file_location)
        
        if duration > 30:
            # 如果音频文件超过30秒，启用分块处理
            result = pipe(file_location, return_timestamps=True,generate_kwargs={"language": "zh"})
        else:
            # 否则，正常处理
            result = pipe(file_location,generate_kwargs={"language": "zh"})
        
        # 将繁体中文转换为简体中文
        cc=OpenCC('t2s')
        simplified_text=cc.convert(result["text"])
        
        print(simplified_text)
        # 返回成功响应
        return JSONResponse(content={"filename": file.filename, "saved_path": file_location,"result_text": simplified_text}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=18002)
