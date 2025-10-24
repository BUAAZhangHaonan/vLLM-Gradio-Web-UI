# multimodal_utils.py
import base64
import mimetypes
import os
from io import BytesIO
from typing import List, Tuple, Dict, Any


import cv2
from PIL import Image


def encode_image_to_base64(image_path: str) -> str:
    """
    将图片文件编码为 Base64 字符串。
    """
    try:
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.convert("RGB").save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return ""


def get_video_thumbnail(video_path: str, temp_dir: str = "temp_thumbs") -> str:
    """
    从视频文件中提取第一帧作为缩略图，并保存为临时文件。
    """
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    try:
        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()
        cap.release()

        if success:
            thumbnail_filename = f"{os.path.basename(video_path)}.jpg"
            thumbnail_path = os.path.join(temp_dir, thumbnail_filename)
            cv2.imwrite(thumbnail_path, frame)
            return thumbnail_path
        return ""
    except Exception as e:
        print(f"Error generating video thumbnail for {video_path}: {e}")
        return ""


def process_files(
    file_paths: List[str],
    temp_dir: str = "temp_thumbs"
) -> Tuple[List[Dict[str, Any]], str]:
    """
    处理上传的文件列表，为 API 请求和前端显示做准备。
    """
    api_content_parts = []
    display_html = ""

    if file_paths is None:
        return api_content_parts, display_html

    for file_path in file_paths:
        mime_type, _ = mimetypes.guess_type(file_path)
        file_name = os.path.basename(file_path)

        if mime_type:
            if mime_type.startswith('image/'):
                base64_image = encode_image_to_base64(file_path)
                if base64_image:
                    api_content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })
                    display_html += f"<img src='file={file_path}' alt='{file_name}' style='max-height: 250px; max-width: 100%; object-fit: contain;'>"

            elif mime_type.startswith('video/'):
                thumbnail_path = get_video_thumbnail(file_path, temp_dir)
                if thumbnail_path:
                    display_html += f"<div><p style='margin-bottom: 5px;'><b>视频: {file_name}</b></p><img src='file={thumbnail_path}' alt='Video thumbnail' style='max-height: 200px; max-width: 100%; object-fit: contain;'></div>"
                else:
                    display_html += f"<div><i>[无法预览视频: {file_name}]</i></div>"
                api_content_parts.append({
                    "type": "text",
                    "text": f"[用户上传了一个视频: {file_name}。模型无法直接看到此视频。]"
                })

            elif mime_type.startswith('audio/'):
                display_html += f"<div><p style='margin-bottom: 5px;'><b>音频: {file_name}</b></p><audio controls src='file={file_path}'></audio></div>"
                api_content_parts.append({
                    "type": "text",
                    "text": f"[用户上传了一个音频: {file_name}。模型无法直接听到此音频。]"
                })

    return api_content_parts, display_html
