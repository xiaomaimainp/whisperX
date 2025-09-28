from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from fastapi.responses import HTMLResponse, JSONResponse
import torch
import warnings
import os
import tempfile
import uvicorn
import librosa
import subprocess
import urllib.parse
import aiohttp
import asyncio
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
            input[type="url"] {
                width: 100%;
                padding: 12px;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin: 10px 0;
                box-sizing: border-box;
            }
            .tab {
                overflow: hidden;
                border: 1px solid #ccc;
                background-color: #f1f1f1;
                border-radius: 5px 5px 0 0;
            }
            .tab button {
                background-color: inherit;
                float: left;
                border: none;
                outline: none;
                cursor: pointer;
                padding: 14px 16px;
                transition: 0.3s;
                color: black;
            }
            .tab button:hover {
                background-color: #ddd;
            }
            .tab button.active {
                background-color: #007bff;
                color: white;
            }
            .tabcontent {
                display: none;
                padding: 20px;
                border: 1px solid #ccc;
                border-top: none;
                border-radius: 0 0 5px 5px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎤 WhisperX 音频转录服务</h1>
            <p>上传音频或视频文件进行语音转录，支持 WAV、MP3、FLAC、M4A、OGG、MP4、MOV、AVI 等格式</p>
            
            <div class="tab">
                <button class="tablinks active" onclick="openTab(event, 'upload')">上传文件</button>
                <button class="tablinks" onclick="openTab(event, 'url')">URL转录</button>
            </div>
            
            <div id="upload" class="tabcontent" style="display: block;">
                <div class="upload-area">
                    <input type="file" id="audioFile" accept="audio/*,video/*" style="margin-bottom: 15px;">
                    <br>
                    <button id="uploadBtn" onclick="uploadFile()">🚀 开始转录</button>
                </div>
            </div>
            
            <div id="url" class="tabcontent">
                <label for="audioUrl">音频文件URL:</label>
                <input type="url" id="audioUrl" placeholder="https://example.com/audio.mp3">
                <button id="urlTranscribeBtn" onclick="transcribeFromUrl()">🔗 通过URL转录</button>
            </div>
            
            <div id="status" style="margin: 20px 0;"></div>
            
            <div id="result" class="result" style="display: none;">
                <h3>📝 转录结果：</h3>
                <div id="transcription"></div>
                <div id="timestamps"></div>
            </div>
        </div>

        <script>
            function openTab(evt, tabName) {
                var i, tabcontent, tablinks;
                tabcontent = document.getElementsByClassName("tabcontent");
                for (i = 0; i < tabcontent.length; i++) {
                    tabcontent[i].style.display = "none";
                }
                tablinks = document.getElementsByClassName("tablinks");
                for (i = 0; i < tablinks.length; i++) {
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }
                document.getElementById(tabName).style.display = "block";
                evt.currentTarget.className += " active";
            }
            
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
            
            async function transcribeFromUrl() {
                const urlInput = document.getElementById('audioUrl');
                const url = urlInput.value.trim();
                const urlBtn = document.getElementById('urlTranscribeBtn');
                const status = document.getElementById('status');
                const result = document.getElementById('result');
                
                if (!url) {
                    alert('请输入音频文件URL');
                    return;
                }
                
                // 显示上传状态
                urlBtn.disabled = true;
                urlBtn.textContent = '⏳ 转录中...';
                status.innerHTML = '<div class="loading">正在下载和转录音频文件，请稍候...</div>';
                result.style.display = 'none';
                
                try {
                    // 发送POST请求到服务器
                    const formData = new FormData();
                    formData.append('url', url);
                    
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
                    urlBtn.disabled = false;
                    urlBtn.textContent = '🔗 通过URL转录';
                }
            }
        </script>
    </body>
    </html>
    """
    return html_content

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(None), 
    url: str = Form(None)
):
    """转录上传的音频文件或通过URL提供的音频文件"""
    
    if not pipe:
        raise HTTPException(status_code=500, detail="模型未加载")
    
    # 检查是否提供了文件或URL
    if not file and (not url or not url.strip()):
        raise HTTPException(status_code=400, detail="请提供音频文件或URL")
    
    tmp_file_path = None
    
    try:
        # 优先处理上传的文件
        if file and file.filename and len(file.filename.strip()) > 0:
            # 处理上传的文件
            allowed_types = [
                'audio/wav', 'audio/mpeg', 'audio/flac', 'audio/mp4', 'audio/ogg',
                'audio/x-wav', 'audio/vnd.wav', 'audio/aac', 'audio/webm', 'audio/3gpp',
                'audio/3gpp2', 'audio/amr', 'audio/amr-wb'
            ]
            
            # 同时检查文件扩展名
            allowed_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.webm', '.3gp', '.3g2', '.amr']
            
            # 获取文件扩展名
            file_extension = os.path.splitext(file.filename)[1].lower() if file.filename else ''
            
            # 如果没有文件扩展名，尝试从Content-Type推断
            if not file_extension:
                extension_map = {
                    'audio/wav': '.wav',
                    'audio/x-wav': '.wav',
                    'audio/vnd.wav': '.wav',
                    'audio/mpeg': '.mp3',
                    'audio/flac': '.flac',
                    'audio/mp4': '.m4a',
                    'audio/ogg': '.ogg',
                    'audio/aac': '.aac',
                    'audio/webm': '.webm'
                }
                file_extension = extension_map.get(file.content_type, '.tmp')
            
            # 如果仍然没有扩展名，使用默认的.tmp
            if not file_extension:
                file_extension = '.tmp'
            
            # 检查是否为允许的类型
            if file.content_type not in allowed_types and file_extension not in allowed_extensions and file_extension != '.tmp':
                raise HTTPException(status_code=400, detail=f"不支持的音频格式: {file.content_type} (扩展名: {file_extension})")
            
            # 保存临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                content = await file.read()
                if not content:
                    raise HTTPException(status_code=400, detail="上传的文件为空")
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
                
        # 如果没有上传文件，则处理URL
        elif url and isinstance(url, str) and url.strip():
            # 验证URL
            parsed_url = urllib.parse.urlparse(url.strip())
            if not parsed_url.scheme or not parsed_url.netloc:
                raise HTTPException(status_code=400, detail="无效的URL")
            
            # 下载音频文件到临时位置
            async with aiohttp.ClientSession() as session:
                async with session.get(url.strip()) as response:
                    if response.status != 200:
                        raise HTTPException(status_code=400, detail=f"无法下载文件，HTTP状态码: {response.status}")
                    
                    # 获取文件扩展名
                    content_type = response.headers.get('content-type', '').split(';')[0]
                    extension = _get_extension_from_content_type(content_type) or _guess_extension_from_url(url.strip()) or '.tmp'
                    
                    # 保存临时文件
                    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp_file:
                        tmp_file_path = tmp_file.name
                        # 分块读取内容以防大文件
                        async for chunk in response.content.iter_chunked(8192):
                            tmp_file.write(chunk)
        else:
            raise HTTPException(status_code=400, detail="请提供有效的音频文件或URL")
        
        # 验证音频文件是否有效
        try:
            # 使用librosa检查音频文件是否有效
            duration = librosa.get_duration(filename=tmp_file_path)
            print(f"音频文件时长: {duration} 秒")
            if duration <= 0:
                raise ValueError("音频文件时长无效")
        except Exception as e:
            # 如果librosa无法读取，可能是文件损坏或格式不支持
            raise HTTPException(status_code=400, detail=f"音频文件无效或已损坏: {str(e)}")
        
        # 转录音频
        result = pipe(
            tmp_file_path,
            generate_kwargs={"task": "transcribe"},
            return_timestamps=True
        )
        
        return JSONResponse({
            "text": result["text"],
            "chunks": result.get("chunks", [])
        })
        
    except HTTPException:
        # 重新抛出已知的HTTP异常
        raise
    except Exception as e:
        # 提供更详细的错误信息
        error_msg = str(e)
        if "Soundfile is either not in the correct format" in error_msg:
            raise HTTPException(status_code=400, detail="音频文件格式不支持或已损坏。请确保文件是有效的音频文件（WAV、MP3、FLAC、M4A等）且未损坏。")
        else:
            raise HTTPException(status_code=500, detail=f"转录失败: {error_msg}")
    finally:
        # 清理临时文件
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except:
                pass

def _get_extension_from_content_type(content_type):
    """根据content-type获取文件扩展名"""
    mime_to_ext = {
        'audio/wav': '.wav',
        'audio/x-wav': '.wav',
        'audio/vnd.wav': '.wav',
        'audio/mpeg': '.mp3',
        'audio/mp3': '.mp3',
        'audio/flac': '.flac',
        'audio/mp4': '.m4a',
        'audio/m4a': '.m4a',
        'audio/x-m4a': '.m4a',
        'audio/ogg': '.ogg',
        'audio/vorbis': '.ogg',
        'audio/aac': '.aac',
        'audio/webm': '.webm',
        'video/mp4': '.mp4',
        'video/webm': '.webm',
        'video/3gpp': '.3gp',
        'video/3gpp2': '.3g2'
    }
    return mime_to_ext.get(content_type.lower())

def _guess_extension_from_url(url):
    """从URL猜测文件扩展名"""
    path = urllib.parse.urlparse(url).path
    return os.path.splitext(path)[1].lower() if '.' in path.split('/')[-1] else None

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "model_loaded": pipe is not None}

if __name__ == "__main__":
    print("🚀 启动 WhisperX Web 转录服务...")
    uvicorn.run(app, host="0.0.0.0", port=7612)