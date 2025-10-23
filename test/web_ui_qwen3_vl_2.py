#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL 前端可视化(复用已有 vLLM OpenAI 兼容服务)
- 不在本进程内加载模型;仅调用 HTTP 接口
- 支持图片(data:URI),视频(推荐 URL 直链)
- 与 web_demo_mm.py 的交互风格类似: Chatbot、多轮、上传、重试、清空

用法示例:
python web_ui_openai_2.py --openai-base-url http://172.17.43.70:8080/v1 --openai-api-key g203 --model qwen3-omni-30b-a3b-thinking-8bit --ui-host 0.0.0.0 --ui-port 7860
python web_ui_openai_2.py --openai-base-url http://172.17.43.70:8000/v1 --openai-api-key g203 --model qwen3-vl-30b-a3b-thinking --ui-host 0.0.0.0 --ui-port 7860
"""

import os
import base64
import mimetypes
from urllib.parse import quote
from typing import List, Dict, Any, Optional

import gradio as gr
from openai import OpenAI
from argparse import ArgumentParser


# ----------------------------
# CLI 参数
# ----------------------------
def get_args():
    p = ArgumentParser()
    p.add_argument("--openai-base-url", type=str, default="http://172.17.43.70:8000/v1",
                   help="现有 vLLM 服务的 OpenAI 兼容 base_url,例如 http://IP:PORT/v1")
    p.add_argument("--openai-api-key", type=str, default="g203",
                   help="vLLM 服务设置的 --api-key")
    p.add_argument("--model", type=str, default="qwen3-vl-30b-a3b-thinking",
                   help="服务端暴露的模型名(/v1/models 返回的 id)")

    # 前端 UI 暴露
    p.add_argument("--ui-host", type=str, default="0.0.0.0",
                   help="Gradio 监听地址,0.0.0.0 便于局域网访问 / VSCode 端口转发")
    p.add_argument("--ui-port", type=int, default=7860,
                   help="优先尝试的端口;占用时将自动回退为任意可用端口")
    p.add_argument("--share", action="store_true",
                   help="Gradio share(一般内网不需要)")

    # 让 vLLM 服务端能拉到你上传的视频(可选)
    p.add_argument("--public-file-base", type=str, default=None,
                   help="将本地上传文件映射为服务端可访问的 URL 前缀,例如 http://IP:PORT/file= ;用于视频/大文件")

    # 采样与限长
    p.add_argument("--max-tokens", type=int, default=8192)
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
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "application/octet-stream"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def to_public_url(abs_path: str, public_file_base: str) -> str:
    return f"{public_file_base}{quote(abs_path)}"


def make_user_content(text: Optional[str],
                      files: List[str],
                      public_file_base: Optional[str]) -> List[Dict[str, Any]]:
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
                content.append(
                    {"type": "input_video", "input_video": {"url": to_data_uri(ap)}})
        else:
            content.append(
                {"type": "text", "text": f"[Unsupported file type: {os.path.basename(ap)}]"})
    return content


def _ensure_local_no_proxy(extra_hosts: Optional[List[str]] = None) -> None:
    def _split_hosts(s: str) -> List[str]:
        s = s.replace(";", ",")
        return [x.strip() for x in s.split(",") if x.strip()]
    hosts = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
    if extra_hosts:
        hosts.update(h for h in extra_hosts if h)
    cur = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    items = set(_split_hosts(cur))
    items.update(hosts)
    merged = ",".join(sorted(items))
    os.environ["NO_PROXY"] = merged
    os.environ["no_proxy"] = merged


def _prepare_gradio_env():
    # 避免被外部环境钉死端口
    os.environ.pop("GRADIO_SERVER_PORT", None)
    # 放大端口扫描范围(若未显式设置)
    os.environ.setdefault("GRADIO_NUM_PORTS", "200")


# ----------------------------
# 核心: 创建 Gradio App
# ----------------------------
def build_app(client: OpenAI, model_name: str, max_tokens: int, temperature: float, top_p: float,
              public_file_base: Optional[str]) -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown(
            "<p align='center'><img src='https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-VL/qwen3vllogo.png' style='height: 72px'/></p>"
        )
        gr.Markdown("<center><h1>Qwen3-VL (via vLLM OpenAI API)</h1></center>")
        gr.Markdown("<center>本前端不加载模型,直接连接已有 vLLM 服务。</center>")

        # 使用 messages 消除弃用告警
        chatbot = gr.Chatbot(label="Qwen3-VL", height=520, type="messages")
        query = gr.Textbox(lines=2, label="输入文本")
        state_backend_history = gr.State(value=[])  # OpenAI 兼容 messages
        state_pending_files = gr.State(value=[])    # List[str]

        with gr.Row():
            upload_btn = gr.UploadButton(
                "📁 上传图片/视频", file_types=["image", "video"])
            submit_btn = gr.Button("🚀 发送")
            regen_btn = gr.Button("🤔 重试")
            clear_btn = gr.Button("🧹 清空")

        def _display_stub(text: str, pending_files: List[str]) -> str:
            files = pending_files or []
            ni = sum(1 for p in files if is_image(p))
            nv = sum(1 for p in files if is_video(p))
            tag = []
            if ni:
                tag.append(f"[图片×{ni}]")
            if nv:
                tag.append(f"[视频×{nv}]")
            prefix = " ".join(tag) + (" " if tag else "")
            return (prefix + text.strip()) if (text and text.strip()) else (prefix or "[媒体消息]")

        # 事件函数
        def on_upload(pending_files: List[str], f) -> List[str]:
            files = list(pending_files or [])
            files.append(f.name)
            return files

        def on_submit(chat: List[Dict[str, Any]],
                      backend_history: List[Dict[str, Any]],
                      pending_files: List[str],
                      text: str):
            user_content = make_user_content(
                text, pending_files or [], public_file_base)
            if not user_content:
                return chat, backend_history, pending_files, gr.update(value="")

            backend_history = list(backend_history or [])
            backend_history.append({"role": "user", "content": user_content})

            # UI 先占位 (messages 结构)
            chat = list(chat or [])
            chat.append(
                {"role": "user", "content": _display_stub(text or "", pending_files)})
            chat.append({"role": "assistant", "content": ""})

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
                try:
                    delta = chunk.choices[0].delta.content or ""
                except Exception:
                    delta = ""
                if delta:
                    partial += delta
                    chat[-1]["content"] = partial
                    yield chat, backend_history, pending_files, gr.update(value="")

            backend_history.append({"role": "assistant", "content": [
                                   {"type": "text", "text": partial}]})
            pending_files = []
            yield chat, backend_history, pending_files, gr.update(value="")

        def on_regen(chat: List[Dict[str, Any]],
                     backend_history: List[Dict[str, Any]]):
            if not backend_history:
                return chat
            if backend_history[-1]["role"] == "assistant":
                backend_history = backend_history[:-1]
            if chat:
                chat[-1] = {"role": "assistant", "content": ""}

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
                try:
                    delta = chunk.choices[0].delta.content or ""
                except Exception:
                    delta = ""
                if delta:
                    partial += delta
                    chat[-1]["content"] = partial
                    yield chat

            backend_history.append({"role": "assistant", "content": [
                                   {"type": "text", "text": partial}]})
            yield chat

        def on_clear():
            return [], [], []

        # 事件绑定
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
            "<small>提示: 图片用 data:URI 发送最稳;视频建议提供可从模型服务端访问的 URL(同机情况下可用 --public-file-base 指向 http://<IP>:<PORT>/file=)。大视频用 data:URI 会很慢。</small>"
        )
    return demo


# ----------------------------
# 入口
# ----------------------------
def main():
    args = get_args()
    _ensure_local_no_proxy([args.ui_host])
    _prepare_gradio_env()

    client = OpenAI(api_key=args.openai_api_key, base_url=args.openai_base_url)
    demo = build_app(
        client=client,
        model_name=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        public_file_base=args.public_file_base,
    )

    # 端口策略：先尝试用户指定；失败(端口冲突)则自动回落为 None(任意可用端口)
    preferred_port = args.ui_port if args.ui_port and args.ui_port > 0 else None
    try:
        demo.queue().launch(
            server_name=args.ui_host,
            server_port=preferred_port,
            share=args.share,
            inbrowser=False,
        )
    except OSError as e:
        msg = str(e)
        if "Cannot find empty port" in msg:
            print("[WARN] 指定端口不可用，自动回退到任意可用端口。")
            demo.queue().launch(
                server_name=args.ui_host,
                server_port=None,   # 让 Gradio 自己找可用端口(受 GRADIO_NUM_PORTS 控制)
                share=args.share,
                inbrowser=False,
            )
        else:
            raise


if __name__ == "__main__":
    main()
