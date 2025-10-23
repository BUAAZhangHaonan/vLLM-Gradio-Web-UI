#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态处理工具
"""

import os
import base64
import mimetypes
from urllib.parse import quote
from typing import List, Dict, Any, Tuple, Optional, Union
from config import IMAGE_EXTS, VIDEO_EXTS, AUDIO_EXTS, MIME_TYPE_MAP


def is_image(path: str) -> bool:
    """判断是否为图像文件"""
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def is_video(path: str) -> bool:
    """判断是否为视频文件"""
    return os.path.splitext(path)[1].lower() in VIDEO_EXTS


def is_audio(path: str) -> bool:
    """判断是否为音频文件"""
    return os.path.splitext(path)[1].lower() in AUDIO_EXTS


def get_media_type(path: str) -> Optional[str]:
    """获取文件的媒体类型"""
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTS:
        return "image"
    elif ext in VIDEO_EXTS:
        return "video"
    elif ext in AUDIO_EXTS:
        return "audio"
    return None


def to_data_uri(path: str) -> str:
    """将本地文件转换为URI格式"""
    mime_type = MIME_TYPE_MAP.get(os.path.splitext(path)[1].lower())
    if mime_type is None:
        mime_type, _ = mimetypes.guess_type(path)
        if mime_type is None:
            mime_type = "application/octet-stream"
    
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def to_public_url(abs_path: str, public_file_base: str) -> str:
    """构造让vLLM服务端可拉取的URL"""
    return f"{public_file_base}{quote(abs_path)}"


def make_user_content(text: Optional[str],
                      files: List[str],
                      public_file_base: Optional[str],
                      model_type: str = "qwen3-omni") -> List[Dict[str, Any]]:
    """
    组装OpenAI Chat Completions的content，支持多模态
    """
    content: List[Dict[str, Any]] = []
    
    if text and text.strip():
        content.append({"type": "text", "text": text.strip()})

    for path in files:
        ap = os.path.abspath(path)
        media_type = get_media_type(ap)
        
        if media_type == "image":
            # 图片优先使用URI
            content.append({
                "type": "image_url", 
                "image_url": {"url": to_data_uri(ap)}
            })
        elif media_type == "video":
            # 修复：由于vLLM目前不支持input_video格式，我们使用文本描述作为临时解决方案
            video_url = to_public_url(ap, public_file_base) if public_file_base else to_data_uri(ap)
            content.append({
                "type": "text", 
                "text": f"[视频: {os.path.basename(ap)}] URL: {video_url}"
            })
        elif media_type == "audio":
            # 音频优先使用URL，否则使用URI
            if public_file_base:
                content.append({
                    "type": "input_audio", 
                    "input_audio": {"url": to_public_url(ap, public_file_base)}
                })
            else:
                content.append({
                    "type": "input_audio", 
                    "input_audio": {"url": to_data_uri(ap)}
                })
        else:
            # 其他文件类型作为文本提示
            content.append({
                "type": "text", 
                "text": f"[Unsupported file type: {os.path.basename(ap)}]"
            })
    
    return content


def parse_assistant_response(response_text: str) -> List[Dict[str, Any]]:
    """
    解析助手的响应，提取文本和音频内容
    """
    # Qwen3-Omni的响应可能包含特殊格式，需要解析
    # 这里假设响应格式为纯文本或包含音频标记的文本
    result = []
    
    # 简单的音频内容检测（实际可能需要更复杂的解析逻辑）
    if "[audio:" in response_text and "]" in response_text:
        # 如果响应中包含音频标记，分离文本和音频
        import re
        audio_pattern = r'\[audio:(.*?)\]'
        matches = re.findall(audio_pattern, response_text)
        
        if matches:
            # 移除音频标记，保留纯文本
            pure_text = re.sub(audio_pattern, '', response_text).strip()
            if pure_text:
                result.append({"type": "text", "text": pure_text})
            
            # 添加音频内容
            for audio_path in matches:
                result.append({"type": "audio", "path": audio_path})
        else:
            result.append({"type": "text", "text": response_text})
    else:
        result.append({"type": "text", "text": response_text})
    
    return result


def format_chat_message(role: str, content: Union[str, List[Dict[str, Any]]], 
                       show_media: bool = True) -> Tuple[str, List[str]]:
    """
    格式化聊天消息，返回显示文本和媒体文件列表
    """
    if isinstance(content, str):
        return content, []
    elif isinstance(content, list):
        text_parts = []
        media_files = []
        
        for item in content:
            if item["type"] == "text":
                text_parts.append(item["text"])
            elif item["type"] in ["image_url", "image"]:
                if show_media and "url" in item.get("image_url", {}):
                    media_files.append(item["image_url"]["url"])
                text_parts.append("[图片]")
            elif item["type"] in ["input_video", "video"]:
                if show_media and "url" in item.get("input_video", {}):
                    media_files.append(item["input_video"]["url"])
                text_parts.append("[视频]")
            elif item["type"] in ["input_audio", "audio"]:
                if show_media and "url" in item.get("input_audio", {}):
                    media_files.append(item["input_audio"]["url"])
                text_parts.append("[音频]")
        
        return " ".join(text_parts), media_files
    else:
        return str(content), []


def detect_model_type(model_name: str) -> str:
    """
    根据模型名称检测模型类型
    """
    model_name_lower = model_name.lower()
    if "omni" in model_name_lower:
        return "qwen3-omni"
    elif "vl" in model_name_lower:
        return "qwen3-vl"
    else:
        return "qwen3-30b"
