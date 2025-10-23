#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-Omni 前端可视化(复用已有 vLLM OpenAI 兼容服务)
- 不在本进程内加载模型;仅调用 HTTP 接口
- 支持图片(data:URI),视频(推荐 URL 直链)
- 与 web_demo_mm.py 的交互风格类似: Chatbot、多轮、上传、重试、清空

python qwen3_vl_ui_openai.py \
  --openai-base-url http://172.17.43.70:8000/v1 \
  --openai-api-key g203 \
  --model qwen3-omni-30b-a3b-thinking \
  --ui-host 0.0.0.0 \
  --ui-port 7860
  
python3 web_ui_qwen3_omni.py --openai-base-url http://172.17.43.70:8000/v1 --openai-api-key g203 --model qwen3-omni --ui-host 0.0.0.0 --ui-port 7862

"""

import os
import base64
import mimetypes
from urllib.parse import quote
from typing import List, Dict, Any, Tuple, Optional

import gradio as gr
from openai import OpenAI
from argparse import ArgumentParser

# ----------------------------
# CLI 参数
# ----------------------------


def get_args():
    p = ArgumentParser()
    # vLLM OpenAI 兼容服务地址与鉴权
    p.add_argument("--openai-base-url", type=str, default="http://172.17.43.70:8000/v1",
                   help="现有 vLLM 服务的 OpenAI 兼容 base_url,例如 http://IP:PORT/v1")
    p.add_argument("--openai-api-key", type=str, default="g203",
                   help="vLLM 服务设置的 --api-key")
    p.add_argument("--model", type=str, default="qwen3-omni-30b-a3b-thinking",
                   help="服务端暴露的模型名(/v1/models 返回的 id);若未设置 served-model-name,可直接填那个很长的路径字符串")

    # 前端 UI 暴露
    p.add_argument("--ui-host", type=str, default="0.0.0.0",
                   help="Gradio 监听地址,0.0.0.0 便于局域网访问 / VSCode 端口转发")
    p.add_argument("--ui-port", type=int, default=7860, help="Gradio 端口")
    p.add_argument("--share", action="store_true",
                   help="Gradio share(一般内网不需要)")

    # 视频直链可选: 让 vLLM 服务端能拉到你上传的视频
    # 例如: --public-file-base http://172.17.43.70:7860/file=
    # 注意: 该 URL 必须对模型服务端可见(同机最稳)
    p.add_argument("--public-file-base", type=str, default=None,
                   help="(可选)将本地上传文件映射为服务端可访问的 URL 前缀,例如 http://IP:PORT/file= ;用于视频/大文件")

    # 采样与限长(按需修改)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)

    return p.parse_args()


# ----------------------------
# 辅助函数
# ----------------------------
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".mpeg"}


def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTS


def to_data_uri(path: str) -> str:
    """把本地文件转为 data:URI(适合图片,视频大文件不推荐)。"""
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        # 默认按二进制流;对图片建议安装合适的扩展名
        mime = "application/octet-stream"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def to_public_url(abs_path: str, public_file_base: str) -> str:
    """
    构造让 vLLM 服务端可拉取的 URL。
    建议 public_file_base 形如 http://<UI_HOST>:<UI_PORT>/file=
    Gradio 会把本地文件映射到 /file=<abs_path> 路由下。
    """
    # 对绝对路径进行 URL 编码
    return f"{public_file_base}{quote(abs_path)}"


def make_user_content(text: Optional[str],
                      files: List[str],
                      public_file_base: Optional[str]) -> List[Dict[str, Any]]:
    """
    组装 OpenAI Chat Completions 的 content(支持文本、图片、视频)。
    - 图片优先用 data:URI,最稳。
    - 视频优先用 URL(需要 public_file_base);否则退化为 data:URI(不推荐,仅小视频)。
    """
    content: List[Dict[str, Any]] = []
    if text and text.strip():
        content.append({"type": "text", "text": text.strip()})

    for path in files:
        ap = os.path.abspath(path)
        if is_image(ap):
            content.append(
                {"type": "image_url", "image_url": {"url": to_data_uri(ap)}})
        elif is_video(ap):
            if public_file_base:
                content.append({"type": "input_video", "input_video": {
                               "url": to_public_url(ap, public_file_base)}})
            else:
                # 兜底方案: 小视频可尝试 data:URI,大视频请提供 public_file_base
                content.append(
                    {"type": "input_video", "input_video": {"url": to_data_uri(ap)}})
        else:
            # 其它文件类型忽略或转为文本提示
            content.append(
                {"type": "text", "text": f"[Unsupported file type: {os.path.basename(ap)}]"})
    return content


# ----------------------------
# 核心: 创建 Gradio App
# ----------------------------
def build_app(client: OpenAI, model_name: str, max_tokens: int, temperature: float, top_p: float,
              public_file_base: Optional[str]) -> gr.Blocks:
    """
    - 维护两份状态: 
      1) backend_history: 真正发给 OpenAI 兼容接口的 messages(list of dict)
      2) pending_files: 还未随用户消息提交的待发送文件列表
    - Chatbot 仅用于展示。
    """

    with gr.Blocks() as demo:
        gr.Markdown(
            "<p align='center'><img src='https://camo.githubusercontent.com/1df0c9499512c1ce1c446481874a712de3eacdb6a5b674c1cd3bf0fafd4280cd/68747470733a2f2f7169616e77656e2d7265732e6f73732d636e2d6265696a696e672e616c6979756e63732e636f6d2f2f5177656e332d4f6d6e692f7177656e335f6f6d6e695f6c6f676f2e706e67' "
            "style='height: 72px'/></p>"
        )
        gr.Markdown("<center><h1>Qwen3-Omni (via vLLM OpenAI API)</h1></center>")
        gr.Markdown("<center>本前端不加载模型,直接连接已有 vLLM 服务。</center>")

        chatbot = gr.Chatbot(label="Qwen3-Omni", height=520)
        query = gr.Textbox(lines=2, label="输入文本")
        # List[{"role":..., "content":[...]}]
        state_backend_history = gr.State(value=[])
        state_pending_files = gr.State(value=[])     # List[str]

        with gr.Row():
            upload_btn = gr.UploadButton(
                "📁 上传图片/视频", file_types=["image", "video"])
            submit_btn = gr.Button("🚀 发送")
            regen_btn = gr.Button("🤔 重试")
            clear_btn = gr.Button("🧹 清空")

        # --- 事件函数 ---
        def on_upload(pending_files: List[str], f) -> List[str]:
            # gr.UploadButton 返回一个 TemporaryFile 对象;取其本地路径
            files = list(pending_files or [])
            files.append(f.name)
            return files

        def on_submit(chat: List[Tuple[str, str]],
                      backend_history: List[Dict[str, Any]],
                      pending_files: List[str],
                      text: str):
            # 1) 组装用户消息
            user_content = make_user_content(
                text, pending_files or [], public_file_base)
            if not user_content:
                # 空输入则无动作
                return chat, backend_history, pending_files, gr.update(value="")

            backend_history = list(backend_history or [])
            backend_history.append({"role": "user", "content": user_content})

            # 2) UI 先占位
            chat = list(chat or [])
            chat.append((text if text else "[媒体消息]", None))

            # 3) 流式请求
            stream = client.chat.completions.create(
                model=model_name,
                messages=backend_history,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
            )

            partial = ""
            for chunk in stream:
                delta = ""
                try:
                    delta = chunk.choices[0].delta.content or ""
                except Exception:
                    delta = ""
                if delta:
                    partial += delta
                    chat[-1] = (chat[-1][0], partial)
                    yield chat, backend_history, pending_files, gr.update(value="")

            # 4) 收尾: 把 assistant 消息放入历史;清空待发送文件与输入框
            backend_history.append({"role": "assistant", "content": [
                                   {"type": "text", "text": partial}]})
            pending_files = []
            yield chat, backend_history, pending_files, gr.update(value="")

        def on_regen(chat: List[Tuple[str, str]],
                     backend_history: List[Dict[str, Any]]):
            # 重试上一个 user 消息: 删除最后一个 assistant,再以历史为 messages 重发
            if not backend_history:
                return chat
            if backend_history[-1]["role"] == "assistant":
                backend_history = backend_history[:-1]

            # UI 占位
            if chat:
                chat[-1] = (chat[-1][0], None)

            stream = client.chat.completions.create(
                model=model_name,
                messages=backend_history,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
            )

            partial = ""
            for chunk in stream:
                delta = ""
                try:
                    delta = chunk.choices[0].delta.content or ""
                except Exception:
                    delta = ""
                if delta:
                    partial += delta
                    chat[-1] = (chat[-1][0], partial)
                    yield chat

            backend_history.append({"role": "assistant", "content": [
                                   {"type": "text", "text": partial}]})
            yield chat

        def on_clear():
            return [], [], []

        # --- 事件绑定 ---
        upload_btn.upload(on_upload, [state_pending_files, upload_btn], [
                          state_pending_files], show_progress=True)
        submit_btn.click(on_submit,
                         [chatbot, state_backend_history,
                             state_pending_files, query],
                         [chatbot, state_backend_history,
                             state_pending_files, query],
                         show_progress=True)
        regen_btn.click(on_regen, [chatbot, state_backend_history], [
                        chatbot], show_progress=True)
        clear_btn.click(on_clear, outputs=[
                        chatbot, state_backend_history, state_pending_files])

        gr.Markdown(
            "<small>提示: 图片用 data:URI 发送最稳;视频建议提供可从模型服务端访问的 URL("
            "同机情况下可用 --public-file-base 指向 http://<IP>:<PORT>/file=)。"
            "大视频用 data:URI 会很慢。</small>"
        )

    return demo


# ----------------------------
# 入口
# ----------------------------
def main():
    args = get_args()
    client = OpenAI(api_key=args.openai_api_key, base_url=args.openai_base_url)

    demo = build_app(
        client=client,
        model_name=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        public_file_base=args.public_file_base,
    )

    demo.queue().launch(
        server_name=args.ui_host,
        server_port=args.ui_port,
        share=args.share,
        inbrowser=False,
    )


if __name__ == "__main__":
    main()
