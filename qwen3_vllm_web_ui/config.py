#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件
"""

# 支持的文件类型
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov",
              ".wmv", ".flv", ".webm", ".mpeg", ".webm"}
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".aac", ".m4a", ".ogg", ".opus", ".wma"}
TEXT_EXTS = {".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".xml"}

# MIME类型映射
MIME_TYPE_MAP = {
    # 图像
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".webp": "image/webp",
    # 视频
    ".mp4": "video/mp4",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".wmv": "video/x-ms-wmv",
    ".flv": "video/x-flv",
    ".webm": "video/webm",
    ".mpeg": "video/mpeg",
    # 音频
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".wma": "audio/x-ms-wma",
}

# 模型类型配置
MODEL_CONFIGS = {
    "qwen3-30b": {
        "supports": {"text"},
        "max_files": 0,
        "description": "纯文本模型"
    },
    "qwen3-vl": {
        "supports": {"text", "image", "video"},
        "max_files": 10,
        "description": "视觉语言模型"
    },
    "qwen3-omni": {
        "supports": {"text", "image", "video", "audio"},
        "max_files": 20,
        "description": "全模态模型"
    }
}

# 默认参数
DEFAULT_PARAMS = {
    "max_tokens": 2048,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 50
}

# UI配置
UI_CONFIG = {
    "title": "Qwen3 MultiModal UI",
    "description": "支持文本、图像、音频、视频的多模态对话界面",
    "height": 600,
    "width": "100%"
}
