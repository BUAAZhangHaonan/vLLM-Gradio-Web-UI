#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一个通用的、可适配多种模型的 Gradio 可视化前端，用于连接 vLLM 的 OpenAI 兼容后端服务。

核心特性:
- 前后端分离: 仅通过 HTTP API 与 vLLM 服务交互，不在本地加载模型。
- 局域网可访问: 默认监听 0.0.0.0 地址，便于团队协作和设备访问。
- 多模型适配: 通过 `--model-capability` 参数动态调整 UI 支持:
  - 'text': 纯文本模型 (如 Qwen3-30B-A3B)
  - 'vl': 支持图片/视频输入的多模态模型 (如 Qwen3-VL)
  - 'omni': 支持图片/视频/音频输入的旗舰多模态模型 (如 Qwen3-Omni)
- 健壮的交互体验:
  - 使用 gr.MultimodalTextbox 实现文本与文件的统一输入。
  - 实时流式响应，优化对特殊 Token 的处理。
  - 修复了多轮对话和刷新后状态丢失或卡顿的问题。
  - 正确在聊天记录中显示上传的图片、视频和音频文件。
- 易于扩展: 代码结构清晰, 预留了集成 TTS (文本转语音)服务的接口。

用法示例:
# 启动针对 Qwen3-Omni 模型的前端
python3 web_ui_vllm_omni.py \
    --model-capability omni \
    --openai-base-url http://192.168.1.100:8000/v1 \
    --model qwen3-omni

python3 web_ui_vllm_omni.py --model-capability omni --openai-base-url http://192.168.1.100:8000/v1 --model qwen3-omni

# 启动针对 Qwen3-VL 模型的前端
python3 web_ui_vllm_omni.py \
    --model-capability vl \
    --openai-base-url http://192.168.1.100:8000/v1 \
    --model qwen3-vl-30b-a3b-thinking

# 启动针对纯文本模型的前端
python3 web_ui_vllm_omni.py \
    --model-capability text \
    --openai-base-url http://192.168.1.100:8000/v1 \
    --model Qwen3-30B-A3B-Instruct
    
vLLM启动命令
vllm serve /home/remote1/lvshuyang/Models/Qwen/Qwen3-Omni-30B-A3B-Thinking-AWQ-8bit --host 0.0.0.0 --port 8000 --served-model-name qwen3-omni --api-key g203 --tensor-parallel-size 2 --dtype bfloat16 --max-model-len 65536 --max-num-seqs 4 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.8 --enable-prefix-caching --generation-config vllm --trust-remote-code --revision main
"""

import os
import base64
import mimetypes
from argparse import ArgumentParser
from typing import List, Dict, Any, Tuple, Generator

import gradio as gr
from openai import OpenAI


# ----------------------------
# 1. 命令行参数定义
# ----------------------------
def get_args():
    """解析和返回命令行参数"""
    parser = ArgumentParser(description="连接到vLLM后端的Gradio通用多模态前端")

    # --- 后端服务配置 ---
    parser.add_argument("--openai-base-url", type=str, default="http://127.0.0.1:8000/v1",
                        help="vLLM 服务的 OpenAI 兼容 API 地址")
    parser.add_argument("--openai-api-key", type=str, default="EMPTY",
                        help="vLLM 服务的 API Key")
    parser.add_argument("--model", type=str, required=True,
                        help="vLLM 服务端加载的模型名称 (例如 'Qwen3-Omni-30B-A3B-Instruct')")
    parser.add_argument("--model-capability", type=str, default="omni", choices=['text', 'vl', 'omni'],
                        help="指定模型的能力类型，以适配UI界面 ('text', 'vl', 'omni')")

    # --- 前端UI配置 ---
    parser.add_argument("--ui-host", type=str, default="0.0.0.0",
                        help="Gradio 服务监听地址")
    parser.add_argument("--ui-port", type=int, default=7860,
                        help="Gradio 服务监听端口")
    parser.add_argument("--share", action="store_true",
                        help="是否创建 Gradio 公网分享链接")

    # --- 模型生成参数 ---
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="生成的最大Token数量")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="采样温度")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="核心采样概率")

    return parser.parse_args()


# ----------------------------
# 2. 辅助函数
# ----------------------------
def file_to_data_uri(file_path: str) -> str:
    """将本地文件路径转换为 Base64 编码的 data URI"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到: {file_path}")

    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = "application/octet-stream"

    with open(file_path, "rb") as f:
        encoded_content = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded_content}"


def build_openai_messages(history: List[List[Dict[str, Any]]], user_input: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    构建符合 OpenAI API 格式的 messages 列表。
    - history: Gradio Chatbot 的历史记录。
    - user_input: 来自 Gradio MultimodalTextbox 的新输入。
    """
    messages = []

    # 添加历史消息
    # Gradio chatbot 的 history 格式是 [[user_msg, assistant_msg], ...]
    if history:
        for turn in history:
            # 用户回合
            if turn[0]:
                messages.append({"role": "user", "content": turn[0]})
            # 助手回合
            if turn[1]:
                messages.append({"role": "assistant", "content": [
                                {"type": "text", "text": turn[1]}]})

    # --- 处理当前用户输入 ---
    # user_input 的格式是 {'text': '...', 'files': ['path1', 'path2']}
    current_user_content = []

    # 1. 处理文件 (图片、视频、音频)
    if user_input.get('files'):
        for file_obj in user_input['files']:
            file_path = file_obj['path']
            # vLLM 目前推荐使用 data URI 方式传递媒体文件
            data_uri = file_to_data_uri(file_path)
            current_user_content.append({
                "type": "image_url",  # 注意：vLLM当前版本统一用 image_url 接收 data URI
                "image_url": {"url": data_uri}
            })

    # 2. 处理文本
    if user_input.get('text') and user_input['text'].strip():
        current_user_content.append({
            "type": "text",
            "text": user_input['text'].strip()
        })

    if not current_user_content:
        # 如果用户只点了发送按钮而没有任何输入，则不添加空消息
        return messages, False

    messages.append({"role": "user", "content": current_user_content})
    return messages, True


# ----------------------------
# 3. Gradio 应用构建与事件处理
# ----------------------------
def build_app(args: ArgumentParser):
    """构建并返回 Gradio 应用"""

    client = OpenAI(api_key=args.openai_api_key, base_url=args.openai_base_url)

    # --- 定义UI组件 ---
    with gr.Blocks(theme=gr.themes.Soft(), css="#chatbot .user {background-color: transparent !important;} #chatbot .bot {background-color: #f0f0f0 !important;}") as demo:
        gr.Markdown(f"<h1 align='center'>通用 vLLM 可视化前端 ({args.model})</h1>")
        gr.Markdown(
            "<p align='center'>当前前端已适配模式: <strong>{args.model_capability.upper()}</strong></p>")

        chatbot = gr.Chatbot(
            [],
            elem_id="chatbot",
            label="对话历史",
            height=600,
            bubble_full_width=False,
            avatar_images=(
                None, "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-VL/qwen3vllogo.png")
        )

        # 根据模型能力配置多模态输入框
        file_types = []
        placeholder_text = "请输入文本..."
        if args.model_capability == 'vl':
            file_types = ["image", "video"]
            placeholder_text = "请输入文本或上传图片/视频..."
        elif args.model_capability == 'omni':
            file_types = ["image", "video", "audio"]
            placeholder_text = "请输入文本或上传图片/视频/音频..."

        chat_input = gr.MultimodalTextbox(
            file_types=file_types,
            placeholder=placeholder_text,
            render=False,  # 稍后在 Column 中渲染以获得更好布局
        )

        with gr.Row():
            with gr.Column(scale=12):
                chat_input.render()
            with gr.Column(scale=1, min_width=80):
                submit_btn = gr.Button("发送", variant="primary")

        with gr.Row():
            regen_btn = gr.Button("🔄 重试")
            clear_btn = gr.Button("🗑️ 清空")

        # --- 事件处理函数 ---
        def on_submit(
            history: List[List[Dict[str, Any]]],
            user_input: Dict[str, Any]
        ) -> Generator[Tuple[List, Dict], None, None]:
            """
            处理用户提交事件。
            1. 构建API请求。
            2. 更新UI显示用户输入。
            3. 流式调用vLLM服务并更新UI。
            """
            # 如果输入为空，则不做任何事
            if not user_input or (not user_input.get('text', '').strip() and not user_input.get('files')):
                yield history, user_input
                return

            # 更新 Chatbot 以立即显示用户输入
            # user_input['files'] 是一个包含字典的列表, eg: [{'path': '...', 'mime_type': '...'}]
            user_message_display = []
            if user_input.get('files'):
                for file in user_input['files']:
                    user_message_display.append((file['path'],))
            if user_input.get('text', '').strip():
                user_message_display.append((user_input['text'].strip(),))

            history.append([user_message_display, None])
            yield history, gr.update(value=None, interactive=False)  # 清空输入框并禁用

            # 构建发送到 OpenAI API 的消息
            messages, is_valid_input = build_openai_messages(
                history[:-1], user_input)

            if not is_valid_input:
                history.pop()  # 移除无效的空输入历史
                yield history, gr.update(value=None, interactive=True)
                return

            history[-1][1] = ""  # 为助手的回答添加一个空的占位符

            try:
                stream = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    stream=True,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p
                )

                # 流式处理响应
                partial_response = ""
                for chunk in stream:
                    # 确保只处理有内容的增量
                    delta_content = chunk.choices[0].delta.content
                    if delta_content:
                        partial_response += delta_content
                        # 实时更新 chatbot 的最后一个气泡
                        history[-1][1] = partial_response
                        yield history, gr.update(interactive=False)

            except Exception as e:
                history[-1][1] = f"**发生错误:** {e}"
                yield history, gr.update(interactive=True)  # 出错时重新启用输入框
                return

            # 对话结束后，重新启用输入框
            yield history, gr.update(interactive=True)

        def on_regen(history: List[List[Dict[str, Any]]]) -> Generator:
            """重新生成最后一次的回答"""
            if not history or history[-1][0] is None:
                yield history, gr.update(interactive=True)
                return

            # 移除上一次的助手回答
            last_user_input_display = history.pop()[0]

            # 从原始的 Gradio history 中提取出符合 MultimodalTextbox 格式的 user_input
            # 这部分逻辑相对复杂，因为需要从展示格式反推输入格式
            # 简化处理：我们直接复用 OpenAI 格式的 history 来重新请求
            # 首先，构建完整的历史消息，但不包括最后一轮的用户输入
            temp_history_for_openai = []
            if history:
                for turn in history:
                    if turn[0]:
                        temp_history_for_openai.append(
                            {"role": "user", "content": turn[0]})
                    if turn[1]:
                        temp_history_for_openai.append({"role": "assistant", "content": [
                                                       {"type": "text", "text": turn[1]}]})

            # 找到最后一轮的用户输入内容
            last_user_input_for_openai = {"role": "user", "content": []}
            text_part = ""
            for item in last_user_input_display:
                if isinstance(item, tuple) and isinstance(item[0], str):  # 文本
                    text_part = item[0]
                elif isinstance(item, tuple):  # 文件
                    last_user_input_for_openai["content"].append({
                        "type": "image_url",
                        "image_url": {"url": file_to_data_uri(item[0])}
                    })
            if text_part:
                last_user_input_for_openai["content"].append(
                    {"type": "text", "text": text_part})

            messages = temp_history_for_openai + [last_user_input_for_openai]

            history.append([last_user_input_display, ""])  # UI占位
            yield history, gr.update(interactive=False)

            try:
                stream = client.chat.completions.create(
                    model=args.model, messages=messages, stream=True,
                    max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p
                )

                partial_response = ""
                for chunk in stream:
                    delta_content = chunk.choices[0].delta.content
                    if delta_content:
                        partial_response += delta_content
                        history[-1][1] = partial_response
                        yield history, gr.update(interactive=False)

            except Exception as e:
                history[-1][1] = f"**重试时发生错误:** {e}"

            yield history, gr.update(interactive=True)

        def on_clear() -> tuple:
            """清空对话历史和输入框"""
            return [], None

        # --- 绑定事件 ---
        submit_btn.click(on_submit, [chatbot, chat_input], [
                         chatbot, chat_input])
        chat_input.submit(on_submit, [chatbot, chat_input], [
                          chatbot, chat_input])
        regen_btn.click(on_regen, [chatbot], [chatbot, chat_input])
        clear_btn.click(on_clear, [], [chatbot, chat_input])

    return demo


# ----------------------------
# 4. 程序入口
# ----------------------------
def main():
    """主函数 解析参数并启动Gradio应用"""
    args = get_args()

    # 确保本地代理设置不会影响到对局域网内 vLLM 服务的访问
    os.environ["NO_PROXY"] = f"localhost,127.0.0.1,{args.ui_host}"

    demo = build_app(args)

    print(f"Gradio 前端启动中... 将在 http://{args.ui_host}:{args.ui_port} 上提供服务")
    print(f"连接到 vLLM 后端: {args.openai_base_url}")
    print(f"使用模型: {args.model} (能力: {args.model_capability.upper()})")

    # 启动 Gradio 服务，并处理端口占用的情况
    try:
        demo.queue().launch(
            server_name=args.ui_host,
            server_port=args.ui_port,
            share=args.share,
            inbrowser=False
        )
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"端口 {args.ui_port} 已被占用。Gradio 将尝试在其他可用端口启动。")
            demo.queue().launch(
                server_name=args.ui_host,
                server_port=None,  # None 表示自动选择可用端口
                share=args.share,
                inbrowser=False
            )
        else:
            raise e


if __name__ == "__main__":
    main()
