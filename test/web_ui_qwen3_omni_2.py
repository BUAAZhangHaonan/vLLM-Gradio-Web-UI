#!/usr/bin-env python3
# -*- coding: utf-8 -*-
"""
Qwen3-Omni 前端可视化(复用已有 vLLM OpenAI 兼容服务)
- 不在本进程内加载模型;仅调用 HTTP 接口
- 支持图片(data:URI),视频(推荐 URL 直链)
- 与 web_demo_mm.py 的交互风格类似: Chatbot、多轮、上传、重试、清空

python3 web_ui_qwen3_omni_final.py --openai-base-url http://172.17.43.70:8000/v1 --openai-api-key g203 --model qwen3-omni --ui-host 0.0.0.0 --ui-port 7862

"""

import os
import base64
import mimetypes
from urllib.parse import quote
from typing import List, Dict, Any, Optional
import json

import gradio as gr
from openai import OpenAI
from argparse import ArgumentParser

# ----------------------------
# CLI 参数 (无变动)
# ----------------------------

def get_args():
    p = ArgumentParser()
    p.add_argument("--openai-base-url", type=str, default="http://172.17.43.70:8000/v1",
                   help="现有 vLLM 服务的 OpenAI 兼容 base_url,例如 http://IP:PORT/v1")
    p.add_argument("--openai-api-key", type=str, default="g203",
                   help="vLLM 服务设置的 --api-key")
    p.add_argument("--model", type=str, default="qwen3-omni",
                   help="服务端暴露的模型名(/v1/models 返回的 id);若未设置 served-model-name,可直接填那个很长的路径字符串")
    p.add_argument("--ui-host", type=str, default="0.0.0.0",
                   help="Gradio 监听地址,0.0.0.0 便于局域网访问 / VSCode 端口转发")
    p.add_argument("--ui-port", type=int, default=7862, help="Gradio 端口")
    p.add_argument("--share", action="store_true",
                   help="Gradio share(一般内网不需要)")
    p.add_argument("--public-file-base", type=str, default=None,
                   help="(可选)将本地上传文件映射为服务端可访问的 URL 前缀,例如 http://IP:PORT/file= ;用于视频/大文件")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    return p.parse_args()

# ----------------------------
# 辅助函数 (无变动)
# ----------------------------
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".mpeg"}

def is_image(path: str) -> bool: return os.path.splitext(path)[1].lower() in IMAGE_EXTS
def is_video(path: str) -> bool: return os.path.splitext(path)[1].lower() in VIDEO_EXTS

def to_data_uri(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime is None: mime = "application/octet-stream"
    with open(path, "rb") as f: b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def to_public_url(abs_path: str, public_file_base: str) -> str:
    return f"{public_file_base}{quote(abs_path)}"

def make_user_content(text: Optional[str], files: List[str], public_file_base: Optional[str]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    if text and text.strip(): content.append({"type": "text", "text": text.strip()})
    for path in files:
        ap = os.path.abspath(path)
        if is_image(ap): content.append({"type": "image_url", "image_url": {"url": to_data_uri(ap)}})
        elif is_video(ap):
            video_url = to_public_url(ap, public_file_base) if public_file_base else to_data_uri(ap)
            content.append({"type": "image_url", "image_url": {"url": video_url}})
        else: content.append({"type": "text", "text": f"[Unsupported file type: {os.path.basename(ap)}]"})
    return content

# ----------------------------
# 核心: 创建 Gradio App (非流式稳定版)
# ----------------------------
def build_app(client: OpenAI, model_name: str, max_tokens: int, temperature: float, top_p: float,
              public_file_base: Optional[str]) -> gr.Blocks:
    
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "<p align='center'><img src='https://camo.githubusercontent.com/1df0c9499512c1ce1c446481874a712de3eacdb6a5b674c1cd3bf0fafd4280cd/68747470733a2f2f7169616e77656e2d7265732e6f73732d636e2d6265696a696e672e616c6979756e63732e636f6d2f2f5177656e332d4f6d6e692f7177656e335f6f6d6e695f6c6f676f2e706e67' "
            "style='height: 72px'/></p>"
        )
        gr.Markdown("<center><h1>Qwen3-Omni (via vLLM OpenAI API)</h1></center>")
        gr.Markdown("<center>本前端不加载模型,直接连接已有 vLLM 服务。</center>")

        chatbot = gr.Chatbot(label="Qwen3-Omni", height=520, type='messages', bubble_full_width=False, show_copy_button=True)
        query = gr.Textbox(lines=2, label="输入文本", placeholder="请输入文本或上传文件...")
        state_pending_files = gr.State(value=[])

        with gr.Row():
            upload_btn = gr.UploadButton("📁 上传图片/视频", file_types=["image", "video"])
            submit_btn = gr.Button("🚀 发送")
            regen_btn = gr.Button("🔄 重试")
            clear_btn = gr.Button("🗑️ 清空")

        def on_upload(pending_files: List[str], f) -> List[str]:
            files = list(pending_files or [])
            files.append(f.name)
            gr.Info(f"已添加文件: {os.path.basename(f.name)}")
            return files

        # 【核心改动】: 将 on_submit 改为普通函数，不再使用 yield
        def on_submit(history: List[Dict[str, any]],
                      pending_files: List[str],
                      text: str):
            
            user_content = make_user_content(text, pending_files or [], public_file_base)
            if not user_content:
                gr.Warning("请输入文本或上传文件！")
                return history, pending_files, text

            history.append({"role": "user", "content": user_content})
            # 立即显示一个“思考中”的占位符
            history.append({"role": "assistant", "content": "🤔..."})
            
            # 立即更新UI，显示用户消息和占位符。这是一个普通 return，会立刻执行。
            # 这是第一步更新
            yield history, [], ""

            messages_to_send = history[:-1] # 发送时不包括占位符
            
            try:
                # 【核心改动】: stream=False，一次性获取完整回复
                response = client.chat.completions.create(
                    model=model_name, messages=messages_to_send, max_tokens=max_tokens,
                    temperature=temperature, top_p=top_p, stream=False,
                )
                
                final_content = response.choices[0].message.content
                history[-1]['content'] = final_content # 替换占位符
            except Exception as e:
                error_msg = f"**错误**: 连接或请求后端服务时出错。\n\n**详情**: {str(e)}"
                history[-1]['content'] = error_msg

            # 返回最终结果，这是第二步更新
            yield history, [], ""


        def on_regen(history: List[Dict[str, any]]):
            if not history or history[-1]['role'] != 'assistant':
                gr.Info("没有可重试的对话。")
                return history
            
            history = history[:-1] # 移除上一条助手回答
            if not history or history[-1]['role'] != 'user': return history
            
            # 复用 on_submit 的逻辑
            yield from on_submit(history, [], "")

        def on_clear():
            return [], []

        # --- 事件绑定 (无变动) ---
        submit_btn.click(
            fn=on_submit,
            inputs=[chatbot, state_pending_files, query],
            outputs=[chatbot, state_pending_files, query]
        )
        query.submit(
            fn=on_submit,
            inputs=[chatbot, state_pending_files, query],
            outputs=[chatbot, state_pending_files, query]
        )
        regen_btn.click(fn=on_regen, inputs=[chatbot], outputs=[chatbot])
        clear_btn.click(fn=on_clear, outputs=[chatbot, state_pending_files])
        upload_btn.upload(fn=on_upload, inputs=[state_pending_files, upload_btn], outputs=[state_pending_files])

        gr.Markdown("<small>提示: ...</small>")
    return demo

# ----------------------------
# 入口 (无变动)
# ----------------------------
def main():
    args = get_args()
    client = OpenAI(api_key=args.openai_api_key, base_url=args.openai_base_url)
    demo = build_app(
        client=client, model_name=args.model, max_tokens=args.max_tokens,
        temperature=args.temperature, top_p=args.top_p, public_file_base=args.public_file_base,
    )
    demo.queue().launch(
        server_name=args.ui_host, server_port=args.ui_port,
        share=args.share, inbrowser=False,
    )

if __name__ == "__main__":
    main()