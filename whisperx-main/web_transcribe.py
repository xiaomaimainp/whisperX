from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
import warnings
import os
import tempfile
import uvicorn
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# 禁用警告
warnings.filterwarnings("ignore")

app = FastAPI(title="WhisperX 音频转录服务")

# 全局变量存储模型
model = None
pipe = None

def load_model():
    """加载 Whisper 模型"""
    global model, pipe
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3-turbo"
    
    print(f"🔧 加载模型: {model_id}")
    print(f"📱 设备: {device}")
    
    # 加载模型
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
    
    print("✅ 模型加载完成!")

@app.on_event("startup")
async def startup_event():
    """应用启动时加载模型"""
    load_model()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """返回上传页面"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>WhisperX 音频转录服务</title>
        <meta charset="utf-8">
        <style>
            body { 
                font-family: Arial, sans-serif; 
                max-width: 800px; 
                margin: 0 auto; 
                padding: 20px; 
                background-color: #f5f5f5;
            }
            .container { 
                background: white; 
                padding: 30px; 
                border-radius: 10px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .upload-area { 
                border: 2px dashed #007bff; 
                padding: 40px; 
                text-align: center; 
                margin: 20px 0; 
                border-radius: 10px;
                background-color: #f8f9fa;
            }
            button { 
                background: #007bff; 
                color: white; 
                padding: 12px 24px; 
                border: none; 
                border-radius: 5px; 
                cursor: pointer; 
                font-size: 16px;
            }
            button:hover { background: #0056b3; }
            button:disabled { background: #6c757d; cursor: not-allowed; }
            .result { 
                margin-top: 20px; 
                padding: 20px; 
                background: #e9ecef; 
                border-radius: 5px; 
                border-left: 4px solid #007bff;
            }
            .loading { color: #007bff; }
            .error { color: #dc3545; }
            .success { color: #28a745; }
            .timestamp { 
                font-family: monospace; 
                background: #f8f9fa; 
                padding: 10px; 
                margin: 10px 0; 
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎤 WhisperX 音频转录服务</h1>
            <p>上传音频文件进行语音转录，支持 WAV、MP3、FLAC、M4A、OGG 等格式</p>
            
            <div class="upload-area">
                <input type="file" id="audioFile" accept="audio/*" style="margin-bottom: 15px;">
                <br>
                <button id="uploadBtn" onclick="uploadFile()">🚀 开始转录</button>
            </div>
            
            <div id="status" style="margin: 20px 0;"></div>
            
            <div id="result" class="result" style="display: none;">
                <h3>📝 转录结果：</h3>
                <div id="transcription"></div>
                <div id="timestamps"></div>
            </div>
        </div>

        <script>
            async function uploadFile() {
                const fileInput = document.getElementById('audioFile');
                const file = fileInput.files[0];
                const uploadBtn = document.getElementById('uploadBtn');
                const status = document.getElementById('status');
                const result = document.getElementById('result');
                
                if (!file) {
                    alert('请选择音频文件');
                    return;
                }
                
                // 显示上传状态
                uploadBtn.disabled = true;
                uploadBtn.textContent = '⏳ 转录中...';
                status.innerHTML = '<div class="loading">正在上传和转录音频文件，请稍候...</div>';
                result.style.display = 'none';
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/transcribe', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        status.innerHTML = '<div class="success">✅ 转录完成!</div>';
                        document.getElementById('transcription').innerHTML = 
                            '<strong>转录文本：</strong><br>' + data.text;
                        
                        // 显示时间戳信息
                        if (data.chunks && data.chunks.length > 0) {
                            let timestampHtml = '<h4>⏰ 时间戳分段：</h4>';
                            data.chunks.forEach((chunk, index) => {
                                const start = chunk.timestamp[0] ? chunk.timestamp[0].toFixed(2) : '0.00';
                                const end = chunk.timestamp[1] ? chunk.timestamp[1].toFixed(2) : '未知';
                                timestampHtml += `<div class="timestamp">[${index + 1}] ${start}s - ${end}s: ${chunk.text.trim()}</div>`;
                            });
                            document.getElementById('timestamps').innerHTML = timestampHtml;
                        }
                        
                        result.style.display = 'block';
                    } else {
                        status.innerHTML = '<div class="error">❌ 转录失败: ' + data.detail + '</div>';
                    }
                } catch (error) {
                    status.innerHTML = '<div class="error">❌ 网络错误: ' + error.message + '</div>';
                } finally {
                    uploadBtn.disabled = false;
                    uploadBtn.textContent = '🚀 开始转录';
                }
            }
        </script>
    </body>
    </html>
    """
    return html_content

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """转录上传的音频文件"""
    
    if not pipe:
        raise HTTPException(status_code=500, detail="模型未加载")
    
    # 检查文件类型
    allowed_types = ['audio/wav', 'audio/mpeg', 'audio/flac', 'audio/mp4', 'audio/ogg']
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="不支持的音频格式")
    
    try:
        # 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # 转录音频
        result = pipe(
            tmp_file_path,
            generate_kwargs={"task": "transcribe"},
            return_timestamps=True
        )
        
        # 清理临时文件
        os.unlink(tmp_file_path)
        
        return JSONResponse({
            "text": result["text"],
            "chunks": result.get("chunks", [])
        })
        
    except Exception as e:
        # 清理临时文件
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        
        raise HTTPException(status_code=500, detail=f"转录失败: {str(e)}")

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "model_loaded": pipe is not None}

if __name__ == "__main__":
    print("🚀 启动 WhisperX Web 转录服务...")
    uvicorn.run(app, host="0.0.0.0", port=7612)