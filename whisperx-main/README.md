# WhisperX 语音转录系统

🎤 基于 OpenAI Whisper Large-v3-Turbo 的高性能语音转录解决方案

## 📋 项目简介

WhisperX 是一个完整的语音识别项目，提供多种转录方式和接口。支持实时语音转文字、批量音频处理、Web 界面上传等功能。适用于会议记录、语音助手、音频内容分析等多种应用场景。

## ✨ 主要特性

- 🚀 **GPU 加速**：支持 CUDA 加速，转录速度快
- 🎯 **高精度识别**：基于 OpenAI Whisper Large-v3-Turbo 模型
- ⏰ **时间戳支持**：提供精确的时间戳分段
- 🌐 **多种接口**：命令行、Web 界面、API 接口
- 📁 **多格式支持**：WAV、MP3、FLAC、M4A、OGG 等
- 🔧 **易于使用**：简单的安装和配置流程

## 🛠️ 安装要求

### 系统要求
- Python 3.8+
- CUDA 支持的 GPU (推荐)
- 8GB+ RAM
- 网络连接 (首次下载模型)

### 依赖安装

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd whisperx-main
   ```

2. **创建虚拟环境**
   ```bash
   conda create -n whisperx python=3.11
   conda activate whisperx
   ```

3. **安装依赖**
   ```bash 
   pip install -r requirements.txt
   ```

## 🚀 快速开始

### 1. 命令行转录 (推荐)

**基础版本：**
```bash
python transcribe_real_audio.py uploads/your_audio.wav
```

**修复版本 (推荐)：**
```bash
python transcribe_fixed.py uploads/your_audio.wav
```

**示例输出：**
```
🎤 WhisperX 实际音频转录 (修复版)
============================================================
📁 音频文件: uploads/beach_description.wav
📱 设备: cuda:0
🔧 模型: openai/whisper-large-v3-turbo

============================================================
📝 转录结果:
============================================================
The beach unfolds like a masterpiece painted by the sun...

============================================================
⏰ 时间戳分段 (已修复):
============================================================
[ 1]   0.00s -   6.26s: The beach unfolds like a masterpiece...
[ 2]   6.82s -  13.00s: soft as crushed pearls beneath bare feet...
[ 3]  13.00s -  18.96s: their crests glowing gold as they catch...

📊 总时长: 40.56 秒
✅ 转录完成!
```

### 2. Web 界面转录

启动 Web 服务：
```bash
python web_transcribe.py
```

然后访问：http://localhost:8000

**Web 界面特性：**
- 📤 拖拽上传音频文件
- 🎵 实时转录进度显示
- 📝 美观的结果展示
- ⏰ 交互式时间戳查看

### 3. 测试和验证

**运行基础测试：**
```bash
python whisper_test.py
```

**运行改进测试：**
```bash
python whisper_test_improved.py
```

## 📁 项目结构

```
whisperx-main/
├── README.md                    # 项目说明文档
├── requirements.txt             # Python 依赖列表
├── upload_audio.py             # 原始上传脚本
├── upload_audio_fixed.py       # 修复版上传脚本
├── whisper_test.py             # 基础测试脚本
├── whisper_test_improved.py    # 改进测试脚本
├── transcribe_real_audio.py    # 命令行转录工具
├── transcribe_fixed.py         # 修复版转录工具 (推荐)
├── web_transcribe.py           # Web 界面服务
├── static/                     # 静态文件目录
└── uploads/                    # 音频文件上传目录
    ├── test_audio.wav          # 测试音频文件
    └── beach_description.wav   # 示例音频文件
```

## 🔧 使用指南

### 支持的音频格式

| 格式 | 扩展名 | 支持状态 |
|------|--------|----------|
| WAV  | .wav   | ✅ 完全支持 |
| MP3  | .mp3   | ✅ 完全支持 |
| FLAC | .flac  | ✅ 完全支持 |
| M4A  | .m4a   | ✅ 完全支持 |
| OGG  | .ogg   | ✅ 完全支持 |

### 转录任务类型

- **transcribe**: 转录为原语言文本
- **translate**: 翻译为英文文本

### 性能优化建议

1. **GPU 使用**：确保 CUDA 可用以获得最佳性能
2. **音频质量**：使用高质量音频文件 (16kHz+ 采样率)
3. **文件大小**：长音频文件会自动分段处理
4. **内存管理**：大文件处理时注意 GPU 内存使用

## 🐛 故障排除

### 常见问题

**1. 模型下载失败**
```bash
# 解决方案：检查网络连接，或使用代理
export HF_ENDPOINT=https://hf-mirror.com
```

**2. CUDA 内存不足**
```bash
# 解决方案：减少 batch_size 或使用 CPU
device = "cpu"  # 在脚本中修改
```

**3. 音频文件格式不支持**
```bash
# 解决方案：使用 ffmpeg 转换格式
ffmpeg -i input.mp4 -ar 16000 output.wav
```

**4. 时间戳重叠问题**
```bash
# 解决方案：使用修复版脚本
python transcribe_fixed.py your_audio.wav
```

### 错误日志

如果遇到问题，请检查：
1. Python 环境和依赖版本
2. CUDA 驱动和 PyTorch 版本兼容性
3. 音频文件完整性和格式
4. 系统内存和 GPU 内存使用情况

## 📊 性能基准

| 音频时长 | GPU (RTX 4090) | CPU (Intel i7) |
|----------|----------------|-----------------|
| 1 分钟   | ~5 秒          | ~30 秒          |
| 10 分钟  | ~30 秒         | ~5 分钟         |
| 1 小时   | ~3 分钟        | ~30 分钟        |

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 发起 Pull Request

## 📄 许可证

本项目基于 MIT 许可证开源。

## 🙏 致谢

- [OpenAI Whisper](https://github.com/openai/whisper) - 核心语音识别模型
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - 模型推理框架
- [FastAPI](https://fastapi.tiangolo.com/) - Web 框架支持

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 GitHub Issue
- 发送邮件至项目维护者

---

**最后更新：** 2025年1月9日  
**版本：** v1.0.0